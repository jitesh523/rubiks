/**
 * Guided Solver Page
 *
 * Real-time camera-based guided solving with move validation
 */

import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import CameraFeed from '../components/CameraFeed';
import Cube3D from '../components/Cube3D';
import MoveValidator from '../components/MoveValidator';
import LoadingSpinner from '../components/LoadingSpinner';
import useWebSocket from '../hooks/useWebSocket';
import './GuidedSolver.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const FACE_NAMES = {
    U: 'White (Up)',
    R: 'Red (Right)',
    F: 'Green (Front)',
    D: 'Yellow (Down)',
    L: 'Orange (Left)',
    B: 'Blue (Back)',
};

function GuidedSolver() {
    const navigate = useNavigate();
    const [sessionId, setSessionId] = useState(null);
    const [mode, setMode] = useState('idle'); // idle, scanning, solving, complete
    const [currentFace, setCurrentFace] = useState('U');
    const [currentMove, setCurrentMove] = useState(null);
    const [instruction, setInstruction] = useState('Initializing...');
    const [detectedColors, setDetectedColors] = useState(null);
    const [avgConfidence, setAvgConfidence] = useState(0);
    const [isScanning, setIsScanning] = useState(false);
    const [validationResult, setValidationResult] = useState(null);
    const [progress, setProgress] = useState(0);
    const [totalSteps, setTotalSteps] = useState(0);
    const [isLoading, setIsLoading] = useState(true);

    const { isConnected, lastMessage, sendMessage } = useWebSocket(sessionId);

    // Grid region for detection (center of camera frame)
    const gridRegion = {
        x: 320,
        y: 180,
        width: 300,
        height: 300,
    };

    // Initialize session
    useEffect(() => {
        const createSession = async () => {
            try {
                const response = await axios.post(`${API_URL}/api/v1/guided/create-session`);
                if (response.data.success) {
                    setSessionId(response.data.session_id);
                    setMode('scanning');
                    setInstruction('Position the white (Up) face in the camera view');
                    setIsLoading(false);
                }
            } catch (error) {
                console.error('Failed to create session:', error);
                setInstruction('Failed to initialize. Please refresh.');
                setIsLoading(false);
            }
        };

        createSession();
    }, []);

    // Handle WebSocket messages
    useEffect(() => {
        if (!lastMessage) return;

        const { type, data } = lastMessage;

        if (type === 'detection_result' && data.success) {
            const { detection } = data;
            setDetectedColors(detection.colors);
            setAvgConfidence(detection.avg_confidence);
            setIsScanning(detection.is_valid);
        } else if (type === 'face_confirmed' && data.success) {
            if (data.all_faces_scanned && data.solution_ready) {
                setMode('solving');
                setInstruction('All faces scanned! Starting guided solving...');
                // Request first instruction
                sendMessage({ type: 'get_instruction' });
            } else if (data.next_face) {
                setCurrentFace(data.next_face);
                setInstruction(`Position the ${FACE_NAMES[data.next_face]} face in the camera view`);
                setDetectedColors(null);
            }
        } else if (type === 'move_validated' && data.success) {
            setValidationResult(data);

            if (data.cube_solved) {
                setMode('complete');
                setInstruction('🎉 Cube Solved! Congratulations!');
            } else {
                // Request next instruction
                setTimeout(() => {
                    sendMessage({ type: 'get_instruction' });
                }, 1000);
            }
        } else if (type === 'instruction' && data.success) {
            setInstruction(data.instruction);

            if (data.mode === 'solving') {
                setMode('solving');
                setCurrentMove(data.current_move);
                setProgress(data.step);
                setTotalSteps(data.total_steps);
            } else if (data.mode === 'complete') {
                setMode('complete');
            }
        } else if (type === 'error') {
            console.error('WebSocket error:', data.error);
        }
    }, [lastMessage, sendMessage]);

    // Handle camera frames
    const handleFrame = useCallback(
        (frameData) => {
            if (!isConnected || mode === 'complete') return;

            sendMessage({
                type: 'process_frame',
                frame_data: frameData,
                grid_region: gridRegion,
            });
        },
        [isConnected, mode, sendMessage]
    );

    const handleConfirmFace = () => {
        if (!detectedColors) return;

        sendMessage({
            type: 'confirm_face',
            face_colors: detectedColors,
        });

        setDetectedColors(null);
        setAvgConfidence(0);
    };

    const handleValidateMove = () => {
        if (!detectedColors) return;

        sendMessage({
            type: 'validate_move',
            current_face_state: detectedColors,
        });
    };

    const handleReset = () => {
        window.location.reload();
    };

    if (isLoading) {
        return (
            <div className="page">
                <div className="container">
                    <LoadingSpinner message="Initializing guided solver..." />
                </div>
            </div>
        );
    }

    return (
        <div className="page">
            <div className="container">
                <div className="page-header">
                    <h1 className="page-title">🎯 Guided Solver</h1>
                    <p className="page-description">
                        Real-time camera-based solving with step-by-step guidance
                    </p>
                </div>

                <div className="guided-solver-layout">
                    {/* Connection Status */}
                    <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
                        <span className="status-dot"></span>
                        <span>{isConnected ? 'Connected' : 'Connecting...'}</span>
                    </div>

                    {/* Current Instruction */}
                    <div className="instruction-panel glass">
                        <h3>📋 Current Instruction</h3>
                        <p className="instruction-text">{instruction}</p>
                        {mode === 'solving' && (
                            <div className="progress-info">
                                <span>Step {progress} of {totalSteps}</span>
                                <div className="progress-bar">
                                    <div
                                        className="progress-fill"
                                        style={{ width: `${(progress / totalSteps) * 100}%` }}
                                    />
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="solver-grid">
                        {/* Camera Feed */}
                        <div className="camera-section">
                            <h3>📷 Camera View</h3>
                            <CameraFeed
                                onFrame={handleFrame}
                                gridRegion={gridRegion}
                                isActive={mode === 'scanning' || mode === 'solving'}
                            />

                            {mode === 'scanning' && detectedColors && (
                                <div className="scan-controls">
                                    <p className="confidence-text">
                                        Confidence: {Math.round(avgConfidence * 100)}%
                                    </p>
                                    <button
                                        className="btn btn-primary"
                                        onClick={handleConfirmFace}
                                        disabled={!isScanning || avgConfidence < 0.7}
                                    >
                                        ✓ Confirm {currentFace} Face
                                    </button>
                                </div>
                            )}

                            {mode === 'solving' && detectedColors && (
                                <div className="scan-controls">
                                    <button
                                        className="btn btn-primary"
                                        onClick={handleValidateMove}
                                    >
                                        ✓ I've Made the Move
                                    </button>
                                </div>
                            )}
                        </div>

                        {/* 3D Visualization */}
                        <div className="cube-section">
                            <h3>🎲 3D Visualization</h3>
                            <Cube3D
                                rotation={[20, 45, 0]}
                                highlightFace={mode === 'scanning' ? currentFace : null}
                            />
                            {mode === 'scanning' && (
                                <div className="face-info">
                                    <p>Scanning: <strong>{FACE_NAMES[currentFace]}</strong></p>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Move Validation Feedback */}
                    {mode === 'solving' && validationResult && (
                        <MoveValidator
                            isValid={validationResult.is_valid}
                            message={validationResult.message}
                            currentMove={currentMove}
                            confidence={avgConfidence}
                        />
                    )}

                    {/* Completion Message */}
                    {mode === 'complete' && (
                        <div className="completion-panel glass">
                            <div className="completion-icon">🎉</div>
                            <h2>Cube Solved!</h2>
                            <p>Congratulations! Your Rubik's Cube is now solved.</p>
                            <div className="completion-actions">
                                <button className="btn btn-primary" onClick={handleReset}>
                                    Solve Another Cube
                                </button>
                                <button
                                    className="btn btn-secondary"
                                    onClick={() => navigate('/')}
                                >
                                    Back to Home
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Controls */}
                    <div className="solver-controls">
                        <button className="btn btn-ghost" onClick={handleReset}>
                            Reset Session
                        </button>
                        <button
                            className="btn btn-secondary"
                            onClick={() => navigate('/scanner')}
                        >
                            Manual Scanner
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default GuidedSolver;
