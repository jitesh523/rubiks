/**
 * Camera Feed Component
 *
 * Manages webcam access and displays live camera feed
 */

import { useEffect, useRef, useState } from 'react';
import './CameraFeed.css';

function CameraFeed({ onFrame, gridRegion, isActive }) {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [hasPermission, setHasPermission] = useState(false);
    const [error, setError] = useState(null);
    const streamRef = useRef(null);
    const intervalRef = useRef(null);

    useEffect(() => {
        const startCamera = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: 'environment',
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    },
                });

                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    streamRef.current = stream;
                    setHasPermission(true);
                    setError(null);
                }
            } catch (err) {
                console.error('Camera access error:', err);
                setError('Failed to access camera. Please grant camera permissions.');
                setHasPermission(false);
            }
        };

        startCamera();

        return () => {
            // Cleanup: stop camera
            if (streamRef.current) {
                streamRef.current.getTracks().forEach((track) => track.stop());
            }
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, []);

    useEffect(() => {
        // Send frames to parent component
        if (hasPermission && isActive && onFrame) {
            intervalRef.current = setInterval(() => {
                captureFrame();
            }, 500); // Capture frame every 500ms

            return () => {
                if (intervalRef.current) {
                    clearInterval(intervalRef.current);
                }
            };
        }
    }, [hasPermission, isActive, onFrame]);

    const captureFrame = () => {
        if (!videoRef.current || !canvasRef.current) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // Set canvas size to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Draw detection grid overlay
        if (gridRegion) {
            ctx.strokeStyle = '#00ff00';
            ctx.lineWidth = 3;
            ctx.strokeRect(gridRegion.x, gridRegion.y, gridRegion.width, gridRegion.height);

            // Draw 3x3 grid
            const cellWidth = gridRegion.width / 3;
            const cellHeight = gridRegion.height / 3;

            ctx.strokeStyle = '#00ff0088';
            ctx.lineWidth = 1;

            for (let i = 1; i < 3; i++) {
                // Vertical lines
                ctx.beginPath();
                ctx.moveTo(gridRegion.x + i * cellWidth, gridRegion.y);
                ctx.lineTo(gridRegion.x + i * cellWidth, gridRegion.y + gridRegion.height);
                ctx.stroke();

                // Horizontal lines
                ctx.beginPath();
                ctx.moveTo(gridRegion.x, gridRegion.y + i * cellHeight);
                ctx.lineTo(gridRegion.x + gridRegion.width, gridRegion.y + i * cellHeight);
                ctx.stroke();
            }
        }

        // Get base64 image data
        const imageData = canvas.toDataURL('image/jpeg', 0.8);

        // Send to parent
        if (onFrame) {
            onFrame(imageData);
        }
    };

    if (error) {
        return (
            <div className="camera-feed-error">
                <div className="error-icon">📷</div>
                <p>{error}</p>
                <button className="btn btn-primary" onClick={() => window.location.reload()}>
                    Retry
                </button>
            </div>
        );
    }

    return (
        <div className="camera-feed">
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="camera-video"
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
        </div>
    );
}

export default CameraFeed;
