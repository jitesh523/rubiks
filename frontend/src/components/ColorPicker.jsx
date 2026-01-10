/**
 * ColorPicker Component
 *
 * Color selection with visual swatches for cube colors
 */

import { COLOR_HEX } from '../utils/cubeUtils';
import './ColorPicker.css';

const COLORS = [
    { name: 'white', label: 'White', key: '1' },
    { name: 'red', label: 'Red', key: '2' },
    { name: 'green', label: 'Green', key: '3' },
    { name: 'yellow', label: 'Yellow', key: '4' },
    { name: 'orange', label: 'Orange', key: '5' },
    { name: 'blue', label: 'Blue', key: '6' },
];

function ColorPicker({ selectedColor, onChange, position = 'bottom' }) {
    const handleColorSelect = (colorName) => {
        onChange(colorName);
    };

    return (
        <div className={`color-picker glass-light ${position}`}>
            <div className="color-swatches">
                {COLORS.map((color) => (
                    <button
                        key={color.name}
                        className={`color-swatch ${selectedColor === color.name ? 'selected' : ''}`}
                        style={{ backgroundColor: COLOR_HEX[color.name] }}
                        onClick={() => handleColorSelect(color.name)}
                        title={`${color.label} (${color.key})`}
                        aria-label={color.label}
                    >
                        {selectedColor === color.name && <span className="checkmark">✓</span>}
                    </button>
                ))}
            </div>
            <div className="color-picker-hint">
                Press 1-6 for quick selection
            </div>
        </div>
    );
}

export default ColorPicker;
