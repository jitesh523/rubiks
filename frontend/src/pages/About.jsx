/**
 * About Page
 *
 * Project information and ML model details
 */

import { useState, useEffect } from 'react';
import { getModelInfo } from '../api/mlApi';
import LoadingSpinner from '../components/LoadingSpinner';
import './About.css';

function About() {
    const [modelInfo, setModelInfo] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchModelInfo();
    }, []);

    const fetchModelInfo = async () => {
        try {
            const info = await getModelInfo();
            setModelInfo(info);
        } catch (err) {
            console.error('Failed to fetch model info:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="page">
            <div className="container">
                <div className="page-header">
                    <h1 className="page-title">About Rubik's Cube Solver</h1>
                    <p className="page-description">
                        AI-powered cube solving with computer vision and machine learning
                    </p>
                </div>

                {/* Project Info */}
                <section className="about-section">
                    <div className="card">
                        <h2>🎯 Project Overview</h2>
                        <p>
                            This is a complete AI-powered Rubik's Cube solver that combines computer vision,
                            machine learning, and the Kociemba algorithm to provide optimal solutions with
                            step-by-step guidance.
                        </p>
                        <p>
                            The system features an interactive web interface built with React, a FastAPI
                            backend for real-time solving, and ML-based color detection for accurate
                            cube state recognition.
                        </p>
                    </div>
                </section>

                {/* Features */}
                <section className="about-section">
                    <h2 className="section-title">✨ Key Features</h2>
                    <div className="features-grid-about">
                        <div className="feature-item glass">
                            <h4>🧠 Kociemba Algorithm</h4>
                            <p>Two-phase algorithm guaranteeing solutions in ≤20 moves</p>
                        </div>
                        <div className="feature-item glass">
                            <h4>🤖 ML Color Detection</h4>
                            <p>KNN classifier with 95%+ accuracy and auto-calibration</p>
                        </div>
                        <div className="feature-item glass">
                            <h4>⚡ FastAPI Backend</h4>
                            <p>High-performance REST API for lightning-fast solutions</p>
                        </div>
                        <div className="feature-item glass">
                            <h4>🎨 Modern UI</h4>
                            <p>React with dark theme, glassmorphism, and responsive design</p>
                        </div>
                    </div>
                </section>

                {/* ML Model Info */}
                <section className="about-section">
                    <div className="card">
                        <h2>🤖 ML Model Status</h2>
                        {loading ? (
                            <LoadingSpinner size="sm" message="Fetching model info..." />
                        ) : modelInfo ? (
                            <div className="model-info-grid">
                                <div className="info-item">
                                    <div className="info-label">Status</div>
                                    <div className={`info-value ${modelInfo.is_trained ? 'success' : 'warning'}`}>
                                        {modelInfo.is_trained ? '✓ Trained' : '⚠ Not Trained'}
                                    </div>
                                </div>
                                {modelInfo.accuracy && (
                                    <div className="info-item">
                                        <div className="info-label">Accuracy</div>
                                        <div className="info-value success">
                                            {(modelInfo.accuracy * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                )}
                                <div className="info-item">
                                    <div className="info-label">Confidence Threshold</div>
                                    <div className="info-value">
                                        {(modelInfo.confidence_threshold * 100).toFixed(0)}%
                                    </div>
                                </div>
                                {modelInfo.last_trained && (
                                    <div className="info-item">
                                        <div className="info-label">Last Trained</div>
                                        <div className="info-value">
                                            {new Date(modelInfo.last_trained).toLocaleDateString()}
                                        </div>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <p className="text-secondary">Model information unavailable</p>
                        )}
                    </div>
                </section>

                {/* How to Use */}
                <section className="about-section">
                    <div className="card">
                        <h2>📖 How to Use</h2>
                        <ol className="usage-steps">
                            <li>
                                <strong>Scan Your Cube:</strong> Go to the Scanner page and input your cube
                                state by clicking on each sticker and selecting its color.
                            </li>
                            <li>
                                <strong>Validate:</strong> Ensure each color appears exactly 9 times on the cube.
                            </li>
                            <li>
                                <strong>Solve:</strong> Click "Solve Cube" to get the optimal solution.
                            </li>
                            <li>
                                <strong>Follow Steps:</strong> Navigate through the solution moves one at a time
                                with explanations for each rotation.
                            </li>
                        </ol>
                    </div>
                </section>

                {/* Auto-Calibration */}
                <section className="about-section">
                    <div className="card highlight-card">
                        <h2>⚡ Auto-Calibration</h2>
                        <p>
                            For best color detection accuracy, calibrate the ML model with your cube and
                            lighting conditions. This requires a solved cube and the Python CLI.
                        </p>
                        <pre className="code-block">
                            <code>python auto_calibrator.py</code>
                        </pre>
                        <p>
                            Follow the on-screen instructions to scan each face. The model trains automatically
                            in ~2 minutes and achieves 95%+ accuracy!
                        </p>
                    </div>
                </section>

                {/* Links */}
                <section className="about-section">
                    <div className="card">
                        <h2>🔗 Links</h2>
                        <div className="links-grid">
                            <a
                                href="https://github.com/jitesh523/rubiks"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="link-item"
                            >
                                <span className="link-icon">📦</span>
                                <div>
                                    <div className="link-title">GitHub Repository</div>
                                    <div className="link-description">View source code</div>
                                </div>
                            </a>
                            <a
                                href="http://localhost:8000/docs"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="link-item"
                            >
                                <span className="link-icon">📚</span>
                                <div>
                                    <div className="link-title">API Documentation</div>
                                    <div className="link-description">Interactive Swagger UI</div>
                                </div>
                            </a>
                        </div>
                    </div>
                </section>

                {/* Tech Stack */}
                <section className="about-section">
                    <h2 className="section-title">🛠️ Technology Stack</h2>
                    <div className="tech-grid">
                        <div className="tech-item">React</div>
                        <div className="tech-item">FastAPI</div>
                        <div className="tech-item">OpenCV</div>
                        <div className="tech-item">scikit-learn</div>
                        <div className="tech-item">Kociemba</div>
                        <div className="tech-item">Axios</div>
                    </div>
                </section>
            </div>
        </div>
    );
}

export default About;
