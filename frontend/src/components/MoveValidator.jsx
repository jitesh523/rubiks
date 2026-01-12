/**
 * Move Validator Component
 *
 * Visual feedback for move validation
 */

import './MoveValidator.css';

function MoveValidator({ isValid, message, currentMove, confidence = 0 }) {
    return (
        <div className={`move-validator ${isValid ? 'valid' : 'invalid'}`}>
            <div className="validator-content">
                <div className="validator-icon">
                    {isValid ? '✓' : '⚠'}
                </div>
                <div className="validator-details">
                    {currentMove && (
                        <div className="current-move">
                            <span className="move-label">Current Move:</span>
                            <span className="move-notation">{currentMove}</span>
                        </div>
                    )}
                    <div className="validator-message">{message}</div>
                    {confidence > 0 && (
                        <div className="confidence-bar">
                            <div className="confidence-label">Confidence</div>
                            <div className="confidence-progress">
                                <div
                                    className="confidence-fill"
                                    style={{ width: `${confidence * 100}%` }}
                                />
                            </div>
                            <div className="confidence-value">{Math.round(confidence * 100)}%</div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default MoveValidator;
