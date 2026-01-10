/**
 * Solver API Service
 *
 * Functions for interacting with the cube solver endpoints
 */

import apiClient from './apiClient';

/**
 * Solve cube from notation string
 * @param {string} cubeString - 54 character string in URFDLB notation
 * @returns {Promise<Object>} Solution response
 */
export const solveFromString = async (cubeString) => {
    const response = await apiClient.post('/api/v1/solver/solve-string', {
        cube_string: cubeString,
    });
    return response.data;
};

/**
 * Solve cube from face arrays
 * @param {Array<Array<string>>} faces - Array of 6 faces, each with 9 colors
 * @param {boolean} useML - Whether to use ML color detection
 * @returns {Promise<Object>} Solution response
 */
export const solveFromFaces = async (faces, useML = false) => {
    const response = await apiClient.post('/api/v1/solver/solve-faces', {
        faces: faces,
        use_ml_detection: useML,
    });
    return response.data;
};

/**
 * Get explanation for a cube move
 * @param {string} move - Move notation (e.g., 'R', "U'", 'F2')
 * @returns {Promise<Object>} Move explanation
 */
export const explainMove = async (move) => {
    const response = await apiClient.get(`/api/v1/solver/explain-move/${move}`);
    return response.data;
};
