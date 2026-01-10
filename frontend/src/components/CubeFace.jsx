/**
 * CubeFace Component
 *
 * Display and interact with a single 3x3 cube face
 */

import { useState } from 'react';
import { COLOR_HEX, getStickerIndex } from '../utils/cubeUtils';
import ColorPicker from './ColorPicker';
import './CubeFace.css';

function CubeFace({ faceName, faceColors, onChange, readonly = false }) {
    const [selectedSticker, setSelectedSticker] = useState(null);

    const handleStickerClick = (index) => {
        if (readonly) return;
        setSelectedSticker(selectedSticker === index ? null : index);
    };

    const handleColorChange = (color) => {
        if (selectedSticker !== null) {
            const newColors = [...faceColors];
            newColors[selectedSticker] = color;
            onChange(newColors);
            setSelectedSticker(null);
        }
    };

    // Close color picker when clicking outside
    const handleClickOutside = () => {
        setSelectedSticker(null);
    };

    return (
        <div className="cube-face-container">
            <div className="cube-face-grid">
                {faceColors.map((color, index) => {
                    const row = Math.floor(index / 3);
                    const col = index % 3;
                    const isCenter = row === 1 && col === 1;

                    return (
                        <div
                            key={index}
                            className={`cube-sticker ${readonly ? 'readonly' : 'interactive'} ${selectedSticker === index ? 'selected' : ''
                                } ${isCenter ? 'center' : ''}`}
                            style={{ backgroundColor: COLOR_HEX[color.toLowerCase()] }}
                            onClick={() => handleStickerClick(index)}
                            role={readonly ? 'presentation' : 'button'}
                            tabIndex={readonly ? -1 : 0}
                            onBlur={() => {
                                // Small delay to allow color picker click to register
                                setTimeout(handleClickOutside, 200);
                            }}
                        >
                            {selectedSticker === index && !readonly && (
                                <ColorPicker
                                    selectedColor={color}
                                    onChange={handleColorChange}
                                    position="bottom"
                                />
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

export default CubeFace;
