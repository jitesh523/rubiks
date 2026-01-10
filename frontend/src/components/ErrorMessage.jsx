/**
 * ErrorMessage Component
 *
 * Display error messages with retry and dismiss functionality
 */

import './ErrorMessage.css';

function ErrorMessage({ message, onRetry, onDismiss }) {
    return (
        <div className="error-message glass">
            <div className="error-content">
                <div className="error-icon">⚠️</div>
                <div className="error-text">
                    <h4>Oops! Something went wrong</h4>
                    <p>{message}</p>
                </div>
            </div>
            <div className="error-actions">
                {onRetry && (
                    <button className="btn btn-primary" onClick={onRetry}>
                        Try Again
                    </button>
                )}
                {onDismiss && (
                    <button className="btn btn-ghost" onClick={onDismiss}>
                        Dismiss
                    </button>
                )}
            </div>
        </div>
    );
}

export default ErrorMessage;
