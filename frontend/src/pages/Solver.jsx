/**
 * Solver Page
 *
 * Display solution with step-by-step navigation
 */

import { useState, useEffect } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import SolutionStep from '../components/SolutionStep';
import CubeFace from '../components/CubeFace';
import LoadingSpinner from '../components/LoadingSpinner';
import { explainMove } from '../api/solverApi';
import './Solver.css';

function Solver() {
    const location = useLocation();
    const navigate = useNavigate();
    const [currentStep, setCurrentStep] = useState(0);
    const [moveExplanations, setMoveExplanations] = useState({});
    const [loadingExplanation, setLoadingExplanation] = useState(false);

    const { faces, solution, moveCount } = location.state || {};

    useEffect(() => {
        if (!solution) {
            // If no solution data, redirect to scanner
            navigate('/scanner');
        }
    }, [solution, navigate]);

    useEffect(() => {
        // Fetch explanation for current move
        if (solution && solution.length > 0 && currentStep < solution.length) {
            const move = solution[currentStep];

            if (!moveExplanations[move]) {
                setLoadingExplanation(true);
                explainMove(move)
                    .then((result) => {
                        setMoveExplanations((prev) => ({
                            ...prev,
                            [move]: result.explanation,
                        }));
                    })
                    .catch((err) => {
                        console.error('Failed to fetch explanation:', err);
                    })
                    .finally(() => {
                        setLoadingExplanation(false);
                    });
            }
        }
    }, [currentStep, solution, moveExplanations]);

    if (!solution) {
        return <LoadingSpinner message="Loading..." />;
    }

    const handlePrevious = () => {
        if (currentStep > 0) {
            setCurrentStep(currentStep - 1);
        }
    };

    const handleNext = () => {
        if (currentStep < solution.length - 1) {
            setCurrentStep(currentStep + 1);
        }
    };

    const progress = solution.length > 0 ? ((currentStep + 1) / solution.length) * 100 : 0;

    return (
        <div className="page">
            <div className="container">
                <div className="page-header">
                    <h1 className="page-title">🧠 Solution</h1>
                    <p className="page-description">
                        {solution.length === 0
                            ? 'Your cube is already solved!'
                            : `Follow these ${solution.length} moves to solve your cube.`}
                    </p>
                </div>

                {solution.length === 0 ? (
                    <div className="solved-message glass">
                        <div className="solved-icon">🎉</div>
                        <h2>Cube Already Solved!</h2>
                        <p>Your cube is in the solved state. No moves needed!</p>
                        <Link to="/scanner" className="btn btn-primary" style={{ marginTop: 'var(--spacing-lg)' }}>
                            Scan Another Cube
                        </Link>
                    </div>
                ) : (
                    <div className="solver-layout">
                        {/* Progress Bar */}
                        <div className="progress-section">
                            <div className="progress-header">
                                <span>Progress</span>
                                <span className="progress-text">
                                    {currentStep + 1} / {solution.length}
                                </span>
                            </div>
                            <div className="progress-bar">
                                <div className="progress-fill" style={{ width: `${progress}%` }}></div>
                            </div>
                        </div>

                        {/* Current Step */}
                        <div className="current-step-section">
                            <SolutionStep
                                move={solution[currentStep]}
                                explanation={moveExplanations[solution[currentStep]] || 'Loading explanation...'}
                                isActive={true}
                                stepNumber={currentStep + 1}
                                totalSteps={solution.length}
                            />
                        </div>

                        {/* Navigation Controls */}
                        <div className="solver-controls">
                            <button
                                className="btn btn-secondary"
                                onClick={handlePrevious}
                                disabled={currentStep === 0}
                            >
                                ← Previous
                            </button>
                            <button className="btn btn-ghost" onClick={() => setCurrentStep(0)}>
                                Reset to Start
                            </button>
                            <button
                                className="btn btn-primary"
                                onClick={handleNext}
                                disabled={currentStep === solution.length - 1}
                            >
                                Next →
                            </button>
                        </div>

                        {/* All Steps Preview */}
                        <div className="all-steps-section">
                            <h3>All Steps</h3>
                            <div className="all-steps-list">
                                {solution.map((move, index) => (
                                    <button
                                        key={index}
                                        className={`step-bubble ${index === currentStep ? 'active' : ''} ${index < currentStep ? 'completed' : ''
                                            }`}
                                        onClick={() => setCurrentStep(index)}
                                    >
                                        <span className="step-bubble-number">{index + 1}</span>
                                        <span className="step-bubble-move">{move}</span>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Cube State Display (Optional) */}
                        {faces && (
                            <div className="cube-state-section">
                                <h3>Original Cube State</h3>
                                <div className="cube-state-grid">
                                    {Object.entries(faces).map(([faceName, faceColors]) => (
                                        <div key={faceName} className="state-face">
                                            <div className="state-face-label">{faceName}</div>
                                            <CubeFace
                                                faceName={faceName}
                                                faceColors={faceColors}
                                                onChange={() => { }}
                                                readonly={true}
                                            />
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* New Scan Button */}
                        <div className="solver-footer">
                            <Link to="/scanner" className="btn btn-secondary">
                                Scan New Cube
                            </Link>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default Solver;
