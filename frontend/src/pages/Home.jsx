/**
 * Home Page
 *
 * Hero section with 3D cube animation and feature showcase
 */

import { Link } from 'react-router-dom';
import './Home.css';

function Home() {
    return (
        <div className="page">
            {/* Hero Section */}
            <section className="hero">
                <div className="container hero-content">
                    <h1 className="hero-title">AI-Powered Rubik's Cube Solver</h1>
                    <p className="hero-description">
                        Scan, solve, and learn with cutting-edge computer vision and machine learning.
                        Get optimal solutions in under 20 moves with step-by-step guidance.
                    </p>

                    {/* 3D Animated Cube */}
                    <div className="hero-cube">
                        <div className="cube-inner">
                            <div className="cube-face front">F</div>
                            <div className="cube-face back">B</div>
                            <div className="cube-face right">R</div>
                            <div className="cube-face left">L</div>
                            <div className="cube-face top">U</div>
                            <div className="cube-face bottom">D</div>
                        </div>
                    </div>

                    <div className="hero-actions">
                        <Link to="/scanner" className="btn btn-primary">
                            Start Scanning 📸
                        </Link>
                        <Link to="/about" className="btn btn-secondary">
                            Learn More
                        </Link>
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section className="features">
                <div className="container">
                    <h2 className="text-center">✨ Amazing Features</h2>
                    <div className="features-grid">
                        <div className="feature-card">
                            <div className="feature-icon">📸</div>
                            <h3 className="feature-title">Interactive Scanner</h3>
                            <p className="feature-description">
                                Manually input your cube state with an intuitive 3×3 grid interface. Visual feedback ensures accuracy.
                            </p>
                        </div>

                        <div className="feature-card">
                            <div className="feature-icon">🧠</div>
                            <h3 className="feature-title">Optimal Solutions</h3>
                            <p className="feature-description">
                                Uses the Kociemba algorithm to find solutions in ≤20 moves. Fast, efficient, and proven.
                            </p>
                        </div>

                        <div className="feature-card">
                            <div className="feature-icon">🤖</div>
                            <h3 className="feature-title">ML Color Detection</h3>
                            <p className="feature-description">
                                Machine learning model achieves 95%+ accuracy. Auto-calibration adapts to your lighting conditions.
                            </p>
                        </div>

                        <div className="feature-card">
                            <div className="feature-icon">🎯</div>
                            <h3 className="feature-title">Step-by-Step Guide</h3>
                            <p className="feature-description">
                                Navigate through solution moves one at a time with clear explanations for each rotation.
                            </p>
                        </div>

                        <div className="feature-card">
                            <div className="feature-icon">⚡</div>
                            <h3 className="feature-title">Real-Time API</h3>
                            <p className="feature-description">
                                FastAPI backend ensures lightning-fast responses. Solve cubes in milliseconds.
                            </p>
                        </div>

                        <div className="feature-card">
                            <div className="feature-icon">🎨</div>
                            <h3 className="feature-title">Beautiful UI</h3>
                            <p className="feature-description">
                                Modern dark theme with glassmorphism effects. Responsive design works on all devices.
                            </p>
                        </div>
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="section">
                <div className="container text-center">
                    <div className="card" style={{ maxWidth: '600px', margin: '0 auto' }}>
                        <h2>Ready to Solve Your Cube?</h2>
                        <p>Start by scanning your scrambled cube and get the solution instantly.</p>
                        <Link to="/scanner" className="btn btn-primary" style={{ marginTop: 'var(--spacing-lg)' }}>
                            Get Started Now →
                        </Link>
                    </div>
                </div>
            </section>
        </div>
    );
}

export default Home;
