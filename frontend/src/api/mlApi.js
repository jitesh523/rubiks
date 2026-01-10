/**
 * ML Model API Service
 *
 * Functions for interacting with machine learning model endpoints
 */

import apiClient from './apiClient';

/**
 * Get ML model information and status
 * @returns {Promise<Object>} Model info including accuracy and training status
 */
export const getModelInfo = async () => {
    const response = await apiClient.get('/api/v1/ml/info');
    return response.data;
};

/**
 * Predict color from RGB values
 * @param {Array<number>} rgb - RGB values [r, g, b]
 * @returns {Promise<Object>} Color prediction with confidence
 */
export const predictColor = async (rgb) => {
    const response = await apiClient.post('/api/v1/ml/predict', {
        rgb: rgb,
    });
    return response.data;
};

/**
 * Train ML model with training data
 * @param {Array<Object>} trainingData - Array of {color, rgb} objects
 * @param {number} confidenceThreshold - Minimum confidence threshold (0-1)
 * @returns {Promise<Object>} Training results
 */
export const trainModel = async (trainingData, confidenceThreshold = 0.7) => {
    const response = await apiClient.post('/api/v1/ml/train', {
        training_data: trainingData,
        confidence_threshold: confidenceThreshold,
    });
    return response.data;
};
