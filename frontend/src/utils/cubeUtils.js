/**
 * Cube Utilities
 *
 * Helper functions for cube state manipulation and validation
 */

/**
 * Color mapping for cube faces
 */
export const FACE_COLORS = {
    U: 'white',
    R: 'red',
    F: 'green',
    D: 'yellow',
    L: 'orange',
    B: 'blue',
};

export const COLOR_HEX = {
    white: '#f8fafc',
    red: '#ef4444',
    green: '#10b981',
    yellow: '#fbbf24',
    orange: '#f97316',
    blue: '#3b82f6',
};

export const FACE_NAMES = {
    U: 'Up (White)',
    R: 'Right (Red)',
    F: 'Front (Green)',
    D: 'Down (Yellow)',
    L: 'Left (Orange)',
    B: 'Back (Blue)',
};

/**
 * Get color name from notation
 * @param {string} notation - Face notation (U, R, F, D, L, B)
 * @returns {string} Color name
 */
export const getColorName = (notation) => {
    return FACE_COLORS[notation] || 'white';
};

/**
 * Get hex color code
 * @param {string} colorName - Color name
 * @returns {string} Hex color code
 */
export const getColorHex = (colorName) => {
    return COLOR_HEX[colorName.toLowerCase()] || COLOR_HEX.white;
};

/**
 * Validate cube state (must have 9 of each color)
 * @param {Object} faces - Object with U, R, F, D, L, B faces
 * @returns {Object} { valid: boolean, error: string }
 */
export const validateCubeState = (faces) => {
    const colorCounts = {
        white: 0,
        red: 0,
        green: 0,
        yellow: 0,
        orange: 0,
        blue: 0,
    };

    // Count colors across all faces
    for (const face of Object.values(faces)) {
        if (!face || face.length !== 9) {
            return {
                valid: false,
                error: 'Each face must have exactly 9 stickers',
            };
        }

        for (const color of face) {
            const colorLower = color.toLowerCase();
            if (colorCounts.hasOwnProperty(colorLower)) {
                colorCounts[colorLower]++;
            } else {
                return {
                    valid: false,
                    error: `Invalid color: ${color}`,
                };
            }
        }
    }

    // Verify each color appears exactly 9 times
    for (const [color, count] of Object.entries(colorCounts)) {
        if (count !== 9) {
            return {
                valid: false,
                error: `Invalid cube: ${color} appears ${count} times (should be 9)`,
            };
        }
    }

    return { valid: true, error: null };
};

/**
 * Convert face objects to 54-character string in URFDLB order
 * @param {Object} faces - Object with U, R, F, D, L, B faces
 * @returns {string} 54-character cube string
 */
export const facesToString = (faces) => {
    const order = ['U', 'R', 'F', 'D', 'L', 'B'];
    let cubeString = '';

    for (const faceName of order) {
        const face = faces[faceName];
        for (const color of face) {
            // Convert color name to notation
            const notation = Object.keys(FACE_COLORS).find(
                (key) => FACE_COLORS[key].toLowerCase() === color.toLowerCase()
            );
            cubeString += notation || 'U';
        }
    }

    return cubeString;
};

/**
 * Convert 54-character string to face objects
 * @param {string} cubeString - 54-character cube string in URFDLB order
 * @returns {Object} Object with U, R, F, D, L, B faces
 */
export const stringToFaces = (cubeString) => {
    if (cubeString.length !== 54) {
        throw new Error('Cube string must be exactly 54 characters');
    }

    const faces = {};
    const order = ['U', 'R', 'F', 'D', 'L', 'B'];

    for (let i = 0; i < 6; i++) {
        const faceName = order[i];
        const start = i * 9;
        const end = start + 9;
        const faceNotation = cubeString.slice(start, end).split('');

        faces[faceName] = faceNotation.map(notation => getColorName(notation));
    }

    return faces;
};

/**
 * Create a solved cube state
 * @returns {Object} Solved cube faces
 */
export const createSolvedCube = () => {
    return {
        U: Array(9).fill('white'),
        R: Array(9).fill('red'),
        F: Array(9).fill('green'),
        D: Array(9).fill('yellow'),
        L: Array(9).fill('orange'),
        B: Array(9).fill('blue'),
    };
};

/**
 * Create an empty cube state (all white)
 * @returns {Object} Empty cube faces
 */
export const createEmptyCube = () => {
    return {
        U: Array(9).fill('white'),
        R: Array(9).fill('white'),
        F: Array(9).fill('white'),
        D: Array(9).fill('white'),
        L: Array(9).fill('white'),
        B: Array(9).fill('white'),
    };
};

/**
 * Get sticker index in a 3x3 grid (0-8, top-left to bottom-right)
 * @param {number} row - Row index (0-2)
 * @param {number} col - Column index (0-2)
 * @returns {number} Sticker index (0-8)
 */
export const getStickerIndex = (row, col) => {
    return row * 3 + col;
};

/**
 * Get row and column from sticker index
 * @param {number} index - Sticker index (0-8)
 * @returns {Object} { row, col }
 */
export const getPosition = (index) => {
    return {
        row: Math.floor(index / 3),
        col: index % 3,
    };
};
