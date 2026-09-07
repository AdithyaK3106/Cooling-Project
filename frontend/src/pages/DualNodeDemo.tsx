// @ts-nocheck
import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html, Line } from '@react-three/drei';
import { useTelemetry } from '../services/telemetryApi';
import * as THREE from 'three';

function Laptop({ position, isMain, risk, fan, mode, connected }) {
  const baseColor = isMain ? '#3b82f6' : (connected ? '#10b981' : '#6b7280'); // Blue for main, Green/Gray for neighbor
  const heatColor = new THREE.Color(baseColor).lerp(new THREE.Color('#ef4444'), risk); // Red based on risk

  return (
    <group position={position}>
      {/* Base */}
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[4, 0.2, 3]} />
        <meshStandardMaterial color={heatColor} />
      </mesh>
      {/* Screen */}
      <mesh position={[0, 1.5, -1.4]} rotation={[-0.2, 0, 0]}>
        <boxGeometry args={[4, 3, 0.2]} />
        <meshStandardMaterial color={heatColor} />
      </mesh>
      {/* Screen Content (Glowing when risk high) */}
      <mesh position={[0, 1.5, -1.29]} rotation={[-0.2, 0, 0]}>
        <planeGeometry args={[3.8, 2.8]} />
        <meshBasicMaterial color={new THREE.Color(0,0,0).lerp(new THREE.Color('#ef4444'), risk * 0.5)} />
      </mesh>
      
      <Html position={[0, 4, 0]} center style={{ pointerEvents: 'none' }}>
        <div className="bg-[#0B0E14]/90 border border-white/20 p-4 rounded-xl shadow-2xl text-sm w-56 text-gray-200 backdrop-blur-xl">
          <div className="font-mono font-bold text-white mb-2 border-b border-white/20 pb-2 flex justify-between items-center">
            <span>{isMain ? 'NODE 1 (MAIN)' : 'NODE 2 (NEIGHBOR)'}</span>
            <span className={`px-2 py-0.5 rounded text-[10px] ${!connected ? 'bg-gray-600' : (mode === 'THERVO' ? 'bg-purple-600' : 'bg-blue-600')}`}>
              {!connected ? 'DISCONNECTED' : mode}
            </span>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-gray-400">Risk Score</span>
            <span className="font-mono font-bold" style={{color: risk > 0.7 ? '#ef4444' : '#10b981'}}>{(risk * 100).toFixed(0)}%</span>
          </div>
          <div className="flex justify-between py-1">
            <span className="text-gray-400">Fan Speed</span>
            <span className="font-mono text-cyan-400">{fan?.toFixed(0) || 0}%</span>
          </div>
        </div>
      </Html>
    </group>
  );
}

function GNNEdge({ p1, p2, active }) {
    const particlesCount = 20;
    const meshRef = useRef();
    
    // Create particles that flow from p1 to p2
    const particles = useMemo(() => {
        const temp = [];
        for(let i=0; i<particlesCount; i++) {
            temp.push({
                offset: i / particlesCount,
                speed: 0.01 + Math.random() * 0.01
            });
        }
        return temp;
    }, []);

    useFrame(() => {
        if (!meshRef.current || !active) return;
        particles.forEach((p, i) => {
            p.offset += p.speed;
            if (p.offset > 1) p.offset = 0;
            const matrix = new THREE.Matrix4();
            const pos = new THREE.Vector3().lerpVectors(p1, p2, p.offset);
            matrix.setPosition(pos);
            meshRef.current.setMatrixAt(i, matrix);
        });
        meshRef.current.instanceMatrix.needsUpdate = true;
    });

    return (
        <group>
            <Line points={[p1, p2]} color={active ? '#a855f7' : '#4b5563'} lineWidth={3} transparent opacity={0.5} />
            {active && (
                <instancedMesh ref={meshRef} args={[null, null, particlesCount]}>
                    <sphereGeometry args={[0.2, 8, 8]} />
                    <meshBasicMaterial color="#ef4444" transparent opacity={0.8} />
                </instancedMesh>
            )}
        </group>
    );
}

export function DualNodeDemo() {
  const { data: telemetry } = useTelemetry();
  
  // Parse telemetry structure from our new brain
  const isLive = telemetry?.live_mode || false;
  
  // Node 1 (Local)
  const risk1 = telemetry?.risk_score || 0.5;
  const fan1 = telemetry?.fan_percent || 0.0;
  const mode1 = telemetry?.risk_level || 'BASELINE';
  
  // Node 2 (Neighbor)
  const connected2 = telemetry?.node2_connected ?? false;
  const risk2 = telemetry?.node2_propagated_risk || 0.0;
  const fan2 = telemetry?.node2_fan || 0.0;
  const mode2 = telemetry?.node2_mode || 'DISCONNECTED';
  
  const p1 = new THREE.Vector3(-6, 0, 0);
  const p2 = new THREE.Vector3(6, 0, 0);
  
  const handleToggle = async () => {
    await fetch('http://localhost:8080/toggle-mode', { method: 'POST' });
  };

  return (
    <div className="flex h-full w-full flex-col">
      <div className="p-6 bg-[#0B0E14] border-b border-white/10 flex justify-between items-center z-10">
        <div>
            <h1 className="text-2xl font-bold text-white">THERVO Multi-Node Orchestration</h1>
            <p className="text-gray-400 text-sm mt-1">Live Personal Hotspot Telemetry</p>
        </div>
        <button 
            onClick={handleToggle}
            className={`px-6 py-3 rounded-lg font-bold transition-all ${isLive ? 'bg-purple-600 hover:bg-purple-500' : 'bg-blue-600 hover:bg-blue-500'}`}
        >
            MODE: {isLive ? 'THERVO (PREDICTIVE GNN)' : 'BASELINE (LOCAL)'}
        </button>
      </div>
      
      <div className="flex-1 relative">
        <Canvas camera={{ position: [0, 8, 15], fov: 45 }}>
          <color attach="background" args={['#111827']} />
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 20, 10]} intensity={1.5} />
          
          <OrbitControls makeDefault maxPolarAngle={Math.PI/2 - 0.1} />
          
          {/* Floor */}
          <gridHelper args={[50, 50, '#374151', '#1f2937']} />
          
          <Laptop position={p1} isMain={true} risk={risk1} fan={fan1} mode={isLive ? 'THERVO' : 'BASELINE'} connected={true} />
          <Laptop position={p2} isMain={false} risk={risk2} fan={fan2} mode={mode2} connected={connected2} />
          
          <GNNEdge p1={new THREE.Vector3(-4, 0, 0)} p2={new THREE.Vector3(4, 0, 0)} active={isLive && connected2} />
          
        </Canvas>
      </div>
      
      {/* Telemetry Footer */}
      <div className="h-48 bg-[#0B0E14] border-t border-white/10 p-6 grid grid-cols-2 gap-8 z-10">
          <div className="bg-white/5 rounded-lg p-4">
              <h3 className="text-sm font-bold text-gray-400 mb-3">NODE 1 (MAIN) RAW METRICS</h3>
              <div className="grid grid-cols-2 gap-4 font-mono text-sm">
                  <div>CPU: <span className="text-white">{telemetry?.cpu?.toFixed(1) || 0}%</span></div>
                  <div>GPU: <span className="text-white">{telemetry?.gpu?.toFixed(1) || 0}%</span></div>
                  <div>CPU Temp: <span className="text-orange-400">{telemetry?.cpu_temp?.toFixed(1) || 0}°C</span></div>
                  <div>GPU Temp: <span className="text-orange-400">{telemetry?.gpu_temp?.toFixed(1) || 0}°C</span></div>
              </div>
          </div>
          <div className="bg-white/5 rounded-lg p-4">
              <h3 className="text-sm font-bold text-gray-400 mb-3">NODE 2 (NEIGHBOR) RAW METRICS</h3>
              {connected2 ? (
                <div className="grid grid-cols-2 gap-4 font-mono text-sm">
                    <div>CPU: <span className="text-white">{telemetry?.node2_cpu?.toFixed(1) || 0}%</span></div>
                    <div>GPU: <span className="text-white">{telemetry?.node2_gpu?.toFixed(1) || 0}%</span></div>
                    <div>CPU Temp: <span className="text-orange-400">{telemetry?.node2_cpu_temp?.toFixed(1) || 0}°C</span></div>
                    <div>Local XGB Risk: <span className="text-cyan-400">{(telemetry?.node2_local_risk * 100)?.toFixed(0) || 0}%</span></div>
                </div>
              ) : (
                  <div className="text-red-500 font-mono mt-4">OFFLINE - WAITING FOR CONNECTION...</div>
              )}
          </div>
      </div>
    </div>
  );
}
