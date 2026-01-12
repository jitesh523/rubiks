/**
 * Custom WebSocket Hook
 *
 * Manages WebSocket connection for guided solver
 */

import { useEffect, useRef, useState, useCallback } from 'react';

const useWebSocket = (sessionId) => {
    const [isConnected, setIsConnected] = useState(false);
    const [lastMessage, setLastMessage] = useState(null);
    const [error, setError] = useState(null);
    const wsRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);

    const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

    const connect = useCallback(() => {
        if (!sessionId) return;

        try {
            const ws = new WebSocket(`${WS_URL}/api/v1/guided/ws/${sessionId}`);

            ws.onopen = () => {
                console.log('WebSocket connected');
                setIsConnected(true);
                setError(null);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    setLastMessage(data);
                } catch (err) {
                    console.error('Failed to parse message:', err);
                }
            };

            ws.onerror = (event) => {
                console.error('WebSocket error:', event);
                setError('WebSocket connection error');
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected');
                setIsConnected(false);

                // Attempt to reconnect after 3 seconds
                reconnectTimeoutRef.current = setTimeout(() => {
                    console.log('Attempting to reconnect...');
                    connect();
                }, 3000);
            };

            wsRef.current = ws;
        } catch (err) {
            console.error('Failed to create WebSocket:', err);
            setError('Failed to connect to server');
        }
    }, [sessionId, WS_URL]);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
        }
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
    }, []);

    const sendMessage = useCallback((message) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(message));
            return true;
        }
        console.warn('WebSocket not connected, cannot send message');
        return false;
    }, []);

    useEffect(() => {
        connect();
        return () => {
            disconnect();
        };
    }, [connect, disconnect]);

    return {
        isConnected,
        lastMessage,
        error,
        sendMessage,
        reconnect: connect,
    };
};

export default useWebSocket;
