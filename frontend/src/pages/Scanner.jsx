/**
 * Scanner Page
 *
 * Interactive cube input with manual color selection
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import CubeFace from '../components/CubeFace';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import { createEmptyCube, FACE_NAMES, validateCubeState, facesToString } from '../utils/cubeUtils';
import { solveFromString } from '../api/solverApi';
import './Scanner.css';

function Scanner() {
    const navigate = useNavigate();
    const [faces, setFaces] = useState(createEmptyCube());
    const [activeFace, setActiveFace] = useState('U');
    const [validation, setValidation] = useState({ valid: false, error: null });
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const faceOrder = ['U', 'R', 'F', 'D', 'L', 'B'];

    // Validate cube on every face change
    useEffect(() => {
        const result = validateCubeState(faces);
        setValidation(result);
    }, [faces]);

    const handleFaceChange = (faceName, newColors) => {
        setFaces({
            ...faces,
            [faceName]: newColors,
        });
    };

    const handleSolveCube = async () => {
        if (!validation.valid) {
            setError(validation.error);
            return;
        }

        setIsLoading(true);
        setError(null);

        try {
            const cubeString = facesToString(faces);
            const result = await solveFromString(cubeString);

            if (result.success) {
                // Navigate to solver page with solution
                navigate('/solver', {
                    state: {
                        faces,
                        solution: result.solution,
                        moveCount: result.move_count,
                    },
                });
            } else {
                setError(result.error || 'Failed to solve cube');
            }
        } catch (err) {
            setError(err.message || 'Failed to connect to solver API');
        } finally {
            setIsLoading(false);
        }
    };

    const handleReset = () => {
        setFaces(createEmptyCube());
        setActiveFace('U');
        setError(null);
    };

    return (
        <div className="page">
            <div className="container">
                <div className="page-header">
                    <h1 className="page-title">🎲 Cube Scanner</h1>
                    <p className="page-description">
                        Input your cube state by clicking on each sticker and selecting its color.
                        Make sure each color appears exactly 9 times.
                    </p>
                </div>

                {error && (
                    <ErrorMessage
                        message={error}
                        onRetry={() => setError(null)}
                        onDismiss={() => setError(null)}
                    />
                )}

                <div className="scanner-layout">
                    {/* Face Tabs */}
                    <div className="face-tabs">
                        {faceOrder.map((faceName) => (
                            <button
                                key={faceName}
                                className={`face-tab ${activeFace === faceName ? 'active' : ''}`}
                                onClick={() => setActiveFace(faceName)}
                            >
                                <span className="face-tab-letter">{faceName}</span>
                                <span className="face-tab-name">{FACE_NAMES[faceName]}</span>
                            </button>
                        ))}
                    </div>

                    {/* Active Face */}
                    <div className="active-face-container">
                        <div className="active-face-header">
                            <h3>{FACE_NAMES[activeFace]}</h3>
                            <p>Click on stickers to change their color</p>
                        </div>
                        <div className="active-face-display">
                            <CubeFace
                                faceName={activeFace}
                                faceColors={faces[activeFace]}
                                onChange={(newColors) => handleFaceChange(activeFace, newColors)}
                            />
                        </div>
                    </div>

                    {/* Validation Status */}
                    <div className="validation-status glass">
                        <h4>Cube Validation</h4>
                        {validation.valid ? (
                            <div className="status-valid">
                                <span className="status-icon">✓</span>
                                <span>Cube state is valid!</span>
                            </div>
                        ) : (
                            <div className="status-invalid">
                                <span className="status-icon">⚠</span>
                                <span>{validation.error || 'Complete all faces'}</span>
                            </div>
                        )}
                    </div>

                    {/* Actions */}
                    <div className="scanner-actions">
                        {isLoading ? (
                            <LoadingSpinner message="Solving your cube..." />
                        ) : (
                            <>
                                <button
                                    className="btn btn-primary"
                                    onClick={handleSolveCube}
                                    disabled={!validation.valid}
                                >
                                    Solve Cube 🧠
                                </button>
                                <button className="btn btn-secondary" onClick={handleReset}>
                                    Reset
                                </button>
                            </>
                        )}
                    </div>
                </div>

                {/* All Faces Preview */}
                <div className="all-faces-preview">
                    <h3>All Faces</h3>
                    <div className="all-faces-grid">
                        {faceOrder.map((faceName) => (
                            <div key={faceName} className="preview-face">
                                <div className="preview-face-label">{faceName}</div>
                                <CubeFace
                                    faceName={faceName}
                                    faceColors={faces[faceName]}
                                    onChange={(newColors) => handleFaceChange(faceName, newColors)}
                                    readonly={false}
                                />
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Scanner;
