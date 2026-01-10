/**
 * LoadingSpinner Component
 *
 * Animated loading indicator with glassmorphism style
 */

import './LoadingSpinner.css';

function LoadingSpinner({ size = 'md', message = 'Loading...' }) {
    const sizeClass = `spinner-${size}`;

    return (
        <div className="loading-spinner-container">
            <div className={`spinner ${sizeClass}`}>
                <div className="spinner-inner"></div>
            </div>
            {message && <p className="spinner-message">{message}</p>}
        </div>
    );
}

export default LoadingSpinner;
