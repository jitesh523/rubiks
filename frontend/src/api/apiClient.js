/**
 * API Client Configuration
 *
 * Centralized axios instance with interceptors for error handling
 */

import axios from 'axios';

// Create axios instance with base configuration
const apiClient = axios.create({
    baseURL: 'http://localhost:8000',
    timeout: 10000,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor
apiClient.interceptors.request.use(
    (config) => {
        // You can add authentication tokens here if needed
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// Response interceptor
apiClient.interceptors.response.use(
    (response) => {
        return response;
    },
    (error) => {
        // Handle common errors
        if (error.response) {
            // Server responded with error status
            console.error('API Error:', error.response.data);

            // You can add custom error handling here
            switch (error.response.status) {
                case 400:
                    error.message = 'Bad Request: ' + (error.response.data.detail || 'Invalid request');
                    break;
                case 404:
                    error.message = 'Not Found: ' + (error.response.data.detail || 'Resource not found');
                    break;
                case 500:
                    error.message = 'Server Error: ' + (error.response.data.detail || 'Internal server error');
                    break;
                default:
                    error.message = error.response.data.detail || 'An error occurred';
            }
        } else if (error.request) {
            // Request was made but no response received
            console.error('Network Error:', error.request);
            error.message = 'Network Error: Unable to connect to the server. Make sure the backend is running.';
        } else {
            // Something else happened
            console.error('Error:', error.message);
        }

        return Promise.reject(error);
    }
);

export default apiClient;
