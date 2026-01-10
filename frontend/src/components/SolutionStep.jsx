/**
 * SolutionStep Component
 *
 * Display a single move in the solution with explanation
 */

import './SolutionStep.css';

function SolutionStep({ move, explanation, isActive, stepNumber, totalSteps }) {
    return (
        <div className={`solution-step ${isActive ? 'active' : ''}`}>
            <div className="step-header">
                <div className="step-number">
                    Step {stepNumber} of {totalSteps}
                </div>
                <div className="step-move">{move}</div>
            </div>
            {explanation && (
                <div className="step-explanation">{explanation}</div>
            )}
            {isActive && (
                <div className="step-indicator">
                    <span className="pulse-dot"></span>
                    Current Step
                </div>
            )}
        </div>
    );
}

export default SolutionStep;
