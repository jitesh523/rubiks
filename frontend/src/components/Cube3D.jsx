/**
 * 3D Rubik's Cube Component
 *
 * Interactive 3D visualization using React Three Fiber
 */

import { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

const FACE_COLORS = {
    U: '#FFFFFF', // White
    D: '#FFFF00', // Yellow
    F: '#00FF00', // Green
    B: '#0000FF', // Blue
    L: '#FFA500', // Orange
    R: '#FF0000', // Red
};

function CubeFaceMesh({ position, color, label }) {
    const meshRef = useRef();

    return (
        <group position={position}>
            <mesh ref={meshRef}>
                <boxGeometry args={[0.95, 0.95, 0.1]} />
                <meshStandardMaterial color={color} />
            </mesh>
            {label && (
                <mesh position={[0, 0, 0.06]}>
                    <planeGeometry args={[0.3, 0.3]} />
                    <meshBasicMaterial color="#000000" transparent opacity={0.5} />
                </mesh>
            )}
        </group>
    );
}

function RubiksCube({ rotation = [0, 0, 0], highlightFace = null }) {
    const groupRef = useRef();

    useFrame(() => {
        if (groupRef.current) {
            groupRef.current.rotation.x = rotation[0];
            groupRef.current.rotation.y = rotation[1];
            groupRef.current.rotation.z = rotation[2];
        }
    });

    // Define cube face positions
    const faceStickerPositions = {
        // Front (Green)
        F: { position: [0, 0, 1.5], rotation: [0, 0, 0] },
        // Back (Blue)
        B: { position: [0, 0, -1.5], rotation: [0, Math.PI, 0] },
        // Up (White)
        U: { position: [0, 1.5, 0], rotation: [-Math.PI / 2, 0, 0] },
        // Down (Yellow)
        D: { position: [0, -1.5, 0], rotation: [Math.PI / 2, 0, 0] },
        // Right (Red)
        R: { position: [1.5, 0, 0], rotation: [0, Math.PI / 2, 0] },
        // Left (Orange)
        L: { position: [-1.5, 0, 0], rotation: [0, -Math.PI / 2, 0] },
    };

    return (
        <group ref={groupRef}>
            {/* Main cube structure */}
            <mesh>
                <boxGeometry args={[3, 3, 3]} />
                <meshStandardMaterial color="#1a1a1a" />
            </mesh>

            {/* Face stickers */}
            {Object.entries(faceStickerPositions).map(([face, { position, rotation }]) => (
                <group key={face} position={position} rotation={rotation}>
                    <CubeFaceMesh
                        position={[0, 0, 0]}
                        color={FACE_COLORS[face]}
                        label={highlightFace === face}
                    />
                </group>
            ))}
        </group>
    );
}

function Cube3D({ rotation = [20, 45, 0], highlightFace = null, animateMove = null }) {
    return (
        <div style={{ width: '100%', height: '400px' }}>
            <Canvas camera={{ position: [5, 5, 5], fov: 50 }}>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} />
                <pointLight position={[-10, -10, -10]} intensity={0.3} />
                <RubiksCube
                    rotation={[rotation[0] * (Math.PI / 180), rotation[1] * (Math.PI / 180), rotation[2] * (Math.PI / 180)]}
                    highlightFace={highlightFace}
                />
                <OrbitControls enableZoom={true} enablePan={false} />
            </Canvas>
        </div>
    );
}

export default Cube3D;
