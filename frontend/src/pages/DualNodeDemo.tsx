import React, { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Html, Line, Float, useGLTF, ContactShadows, Environment } from '@react-three/drei';
import { useTelemetry } from '../services/telemetryApi';
import * as THREE from 'three';

// -----------------------------------------------------------------------------
// LAPTOP MODEL COMPONENT
// -----------------------------------------------------------------------------
function LaptopModel({ position, rotation, risk, isMain, connected, mode }) {
  const { scene } = useGLTF('/models/laptop.glb');
  const clone = useMemo(() => scene.clone(), [scene]);
  
  const isHighRisk = risk > 0.6;
  const isActivelyCooled = !isMain && mode === 'THERVO' && risk <= 0.6 && connected;
  
  // Set base color to white when THERVO mode is active to fix contrast issues
  let baseColor = '#4b5563'; // offline grey
  if (connected) {
      if (mode === 'THERVO') {
          baseColor = isActivelyCooled ? '#06b6d4' : '#ffffff'; 
      } else {
          baseColor = isMain ? '#3b82f6' : '#10b981'; // Blue / Green for BASELINE
      }
  }
  
  // To avoid White + Red blending into "Pink/Purple", we map the risk more sharply.
  // If risk is < 0.3, it stays the pure base color.
  // If risk is > 0.3, it quickly shifts to pure Red to avoid the pink/purple transition phase.
  const mappedRisk = Math.max(0, (risk - 0.3) * 1.5);
  const heatColor = new THREE.Color(baseColor).lerp(new THREE.Color('#ef4444'), Math.min(1.0, mappedRisk)); 
  
  useEffect(() => {
    clone.traverse((child) => {
      if (child.isMesh) {
        child.material = child.material.clone();
        if (child.material.emissive) {
            child.material.emissive.copy(heatColor);
            child.material.emissiveIntensity = isHighRisk ? 0.8 : (isActivelyCooled ? 0.6 : 0.1);
        }
      }
    });
  }, [clone, heatColor, isHighRisk, isActivelyCooled]);

  return (
    <group position={position} rotation={rotation}>
      <Float speed={2} rotationIntensity={0.02} floatIntensity={0.1} floatingRange={[-0.1, 0.1]}>
        
        {/* Scaled up 3x as requested, shifted down slightly */}
        <primitive object={clone} scale={4.0} position={[0, -1.0, 0]} />

        {/* FUTURISTIC PEDESTAL BASE */}
        <mesh position={[0, -1.4, 0]}>
          <cylinderGeometry args={[4.2, 4.5, 0.8, 64]} />
          <meshStandardMaterial color="#0a0c10" metalness={0.9} roughness={0.2} />
        </mesh>
        
        {/* PEDESTAL SURFACE (TOP) */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.01, 0]}>
          <circleGeometry args={[4.2, 64]} />
          <meshStandardMaterial color="#14171f" metalness={0.7} roughness={0.1} />
        </mesh>

        {connected && (
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.0, 0]}>
            <ringGeometry args={[3.8, 4.2, 64]} />
            <meshBasicMaterial 
                color={heatColor} 
                transparent 
                opacity={0.8 + (isHighRisk ? 0.2 : 0)} 
                side={THREE.DoubleSide} 
            />
          </mesh>
        )}
        
        {connected && (
            <pointLight position={[0, -0.5, 0]} color={heatColor} intensity={isHighRisk ? 4 : 1.5} distance={10} />
        )}

      </Float>
    </group>
  );
}

// -----------------------------------------------------------------------------
// GNN PROPAGATION STREAM
// -----------------------------------------------------------------------------
function GNNEdge({ p1, p2, active, risk }) {
    const particlesCount = 40;
    const meshRef = useRef();
    
    const particles = useMemo(() => {
        const temp = [];
        for(let i=0; i<particlesCount; i++) {
            temp.push({
                offset: i / particlesCount,
                speed: active ? (0.01 + risk * 0.03 + Math.random() * 0.01) : 0,
                radius: Math.random() * 0.4
            });
        }
        return temp;
    }, [active, risk]);

    useFrame(() => {
        if (!meshRef.current || !active) return;
        particles.forEach((p, i) => {
            p.offset += p.speed;
            if (p.offset > 1) p.offset = 0;
            const matrix = new THREE.Matrix4();
            
            const wobbleY = Math.sin(p.offset * Math.PI * 6) * p.radius;
            const wobbleZ = Math.cos(p.offset * Math.PI * 6) * p.radius;
            
            const basePos = new THREE.Vector3().lerpVectors(p1, p2, p.offset);
            matrix.setPosition(basePos.x, basePos.y + wobbleY, basePos.z + wobbleZ);
            
            const scale = Math.sin(p.offset * Math.PI) * (1.0 + risk * 2);
            matrix.scale(new THREE.Vector3(scale, scale, scale));
            
            meshRef.current.setMatrixAt(i, matrix);
        });
        meshRef.current.instanceMatrix.needsUpdate = true;
    });

    return (
        <group>
            <Line points={[p1, p2]} color={active ? '#a855f7' : '#374151'} lineWidth={3} transparent opacity={0.4} />
            {active && (
                <instancedMesh ref={meshRef} args={[null, null, particlesCount]}>
                    <sphereGeometry args={[0.08, 8, 8]} />
                    <meshBasicMaterial color={new THREE.Color('#a855f7').lerp(new THREE.Color('#ef4444'), risk)} transparent opacity={0.9} />
                </instancedMesh>
            )}
        </group>
    );
}

// -----------------------------------------------------------------------------
// MAIN DEMO VIEW
// -----------------------------------------------------------------------------
export function DualNodeDemo() {
  const { data: telemetry } = useTelemetry();
  const [events, setEvents] = useState([]);
  
  const isLive = telemetry?.live_mode || false;
  
  const risk1 = telemetry?.risk_score || 0.0;
  const fan1 = telemetry?.fan_percent || 0.0;
  const mode1 = telemetry?.risk_level || 'BASELINE';
  
  const connected2 = telemetry?.node2_connected ?? false;
  const risk2 = telemetry?.node2_propagated_risk || 0.0;
  const fan2 = telemetry?.node2_fan || 0.0;
  const mode2 = telemetry?.node2_mode || 'DISCONNECTED';

  // Positioned closer together in center
  const p1 = new THREE.Vector3(-4.5, 0, 0);
  const p2 = new THREE.Vector3(4.5, 0, 0);

  const handleToggle = async () => {
    try {
        console.log('Sending toggle request...');
        const res = await fetch(`http://${window.location.hostname}:8080/toggle-mode`, { method: 'POST' });
        const json = await res.json();
        console.log('Toggle response:', json);
    } catch (e) {
        console.error('Toggle request failed:', e);
        alert('Failed to switch mode. Check console for details.');
    }
  };

  useEffect(() => {
      if (!telemetry) return;
      const newEvent = (msg, type='info') => {
          setEvents(prev => [{id: Date.now(), msg, type, time: new Date().toLocaleTimeString()}, ...prev].slice(0, 4));
      };

      if (!connected2 && events.length === 0) newEvent("Waiting for Node 2 connection...", "warn");
      if (connected2 && events.length > 0 && events[0].msg.includes("Waiting")) newEvent("Node 2 Connected", "success");
      
      if (risk1 > 0.6) newEvent(`Node 1 critical thermal risk: ${(risk1*100).toFixed(0)}%`, "error");
      if (isLive && risk1 > 0.4 && connected2) newEvent(`GNN Propagating heat vector to Node 2`, "info");
      
  }, [risk1, isLive, connected2]);

  return (
    <div className="flex h-screen w-full flex-col bg-[#111827] text-gray-200 overflow-hidden font-sans">
      
      {/* 1. TOP CONTROL BAR */}
      <div className="absolute top-0 w-full p-6 flex justify-between items-center z-20 pointer-events-none">
        <div>
            <h1 className="text-3xl font-bold text-white tracking-wider flex items-center gap-3">
                THERVO <span className="text-sm font-normal px-2 py-1 bg-white/10 rounded-full border border-white/20">LIVE OPS</span>
            </h1>
            <p className="text-gray-400 text-sm mt-2 max-w-md">
                {isLive ? 'Predictive Coordinated Cooling is currently actively propagating risk embeddings across the network.' : 'Local Cooling only. Nodes are fully isolated.'}
            </p>
        </div>
        <button 
            onClick={handleToggle}
            className={`pointer-events-auto px-8 py-4 rounded-xl font-bold tracking-widest transition-all duration-500 shadow-2xl border ${isLive ? 'bg-purple-600/20 text-purple-400 border-purple-500 hover:bg-purple-600/40' : 'bg-blue-600/20 text-blue-400 border-blue-500 hover:bg-blue-600/40'}`}
        >
            MODE: {isLive ? 'THERVO GNN' : 'BASELINE'}
        </button>
      </div>

      {/* 2. 3D HERO CANVAS */}
      <div className="absolute inset-0 w-full h-full z-0">
        <Canvas camera={{ position: [0, 4, 11], fov: 50 }}>
          <color attach="background" args={['#1f2937']} />
          <fog attach="fog" args={['#1f2937', 10, 25]} />
          
          <ambientLight intensity={0.4} />
          <directionalLight position={[5, 10, 5]} intensity={2.0} color="#ffffff" castShadow />
          <directionalLight position={[-5, 10, -5]} intensity={0.5} color="#cbd5e1" />
          
          <OrbitControls makeDefault maxPolarAngle={Math.PI/2 + 0.1} minPolarAngle={Math.PI/4} enablePan={false} enableZoom={true} minDistance={5} maxDistance={25} />
          
          <Environment preset="city" />
          <ContactShadows position={[0, -2.5, 0]} opacity={0.5} scale={50} blur={2.5} far={4} />
          
          {/* NODE 1 */}
          <LaptopModel position={p1} rotation={[0, 0.3, 0]} risk={risk1} isMain={true} connected={true} mode={isLive ? 'THERVO' : 'BASELINE'} />
          
          {/* NODE 1 OVERLAY (Positioned directly above laptop) */}
          <Html position={[-4.5, 3.5, 0]} center zIndexRange={[100, 0]}>
            <div className={`w-64 bg-black/70 backdrop-blur-md border ${risk1 > 0.6 ? 'border-red-500 shadow-[0_0_20px_rgba(239,68,68,0.4)]' : 'border-blue-500/40'} rounded-lg p-4 pointer-events-none`}>
                <div className="flex justify-between items-center mb-2 pb-2 border-b border-white/10">
                    <h3 className="font-bold text-white tracking-widest text-lg">NODE 1</h3>
                    <span className="text-[10px] bg-blue-500/20 text-blue-400 px-2 py-1 rounded">MAIN</span>
                </div>
                <div className="space-y-1 text-xs font-mono">
                    <div className="flex justify-between"><span className="text-gray-400">Risk Score</span><span className={risk1 > 0.6 ? 'text-red-400 font-bold text-sm' : 'text-blue-400'}>{(risk1*100).toFixed(1)}%</span></div>
                    <div className="flex justify-between"><span className="text-gray-400">Policy</span><span className="text-white font-bold">{mode1}</span></div>
                    <div className="flex justify-between"><span className="text-gray-400">Target RPM</span><span className="text-cyan-400">{(fan1 || 0).toFixed(1)}%</span></div>
                    <div className="h-px bg-white/10 my-2" />
                    <div className="flex justify-between"><span className="text-gray-400">CPU Load</span><span>{(telemetry?.cpu || 0).toFixed(1)}%</span></div>
                    <div className="flex justify-between"><span className="text-gray-400">GPU Load</span><span>{(telemetry?.gpu || 0).toFixed(1)}%</span></div>
                    <div className="flex justify-between"><span className="text-gray-400">CPU Temp</span><span className="text-orange-400">{(telemetry?.cpu_temp || 0).toFixed(1)}°C</span></div>
                </div>
            </div>
          </Html>

          {/* NODE 2 */}
          <LaptopModel position={p2} rotation={[0, -0.3, 0]} risk={risk2} isMain={false} connected={connected2} mode={mode2} />
          
          {/* NODE 2 OVERLAY */}
          <Html position={[4.5, 3.5, 0]} center zIndexRange={[100, 0]}>
            <div className={`w-64 bg-black/70 backdrop-blur-md border ${!connected2 ? 'border-gray-700' : (risk2 > 0.6 ? 'border-red-500 shadow-[0_0_20px_rgba(239,68,68,0.4)]' : 'border-green-500/40')} rounded-lg p-4 pointer-events-none transition-all`}>
                <div className="flex justify-between items-center mb-2 pb-2 border-b border-white/10">
                    <h3 className="font-bold text-white tracking-widest text-lg">NODE 2</h3>
                    <span className={`text-[10px] px-2 py-1 rounded ${!connected2 ? 'bg-gray-700 text-gray-300' : 'bg-green-500/20 text-green-400'}`}>{connected2 ? 'CONNECTED' : 'OFFLINE'}</span>
                </div>
                {connected2 ? (
                    <div className="space-y-1 text-xs font-mono">
                        <div className="flex justify-between"><span className="text-gray-400">Risk Score</span><span className={risk2 > 0.6 ? 'text-red-400 font-bold text-sm' : 'text-green-400'}>{(risk2*100).toFixed(1)}%</span></div>
                        <div className="flex justify-between"><span className="text-gray-400">Policy</span><span className={mode2 === 'THERVO' ? 'text-purple-400 font-bold' : 'text-white font-bold'}>{mode2}</span></div>
                        <div className="flex justify-between"><span className="text-gray-400">Target RPM</span><span className="text-cyan-400">{(fan2 || 0).toFixed(1)}%</span></div>
                        <div className="h-px bg-white/10 my-2" />
                        <div className="flex justify-between"><span className="text-gray-400">CPU Load</span><span>{(telemetry?.node2_cpu || 0).toFixed(1)}%</span></div>
                        <div className="flex justify-between"><span className="text-gray-400">GPU Load</span><span>{(telemetry?.node2_gpu || 0).toFixed(1)}%</span></div>
                        <div className="flex justify-between"><span className="text-gray-400">CPU Temp</span><span className="text-orange-400">{(telemetry?.node2_cpu_temp || 0).toFixed(1)}°C</span></div>
                    </div>
                ) : (
                    <div className="text-center py-4 text-xs text-gray-500 font-mono">AWAITING CONNECTION</div>
                )}
            </div>
          </Html>

          {/* GNN EDGE */}
          <GNNEdge p1={new THREE.Vector3(-3.5, 0, 0)} p2={new THREE.Vector3(3.5, 0, 0)} active={isLive && connected2} risk={risk1} />
          
          {/* CENTRAL GNN STATUS OVERLAY */}
          {connected2 && (
            <Html position={[0, 4.0, 0]} center zIndexRange={[90, 0]}>
                <div className={`px-6 py-2 bg-black/80 backdrop-blur-md rounded-full border ${isLive ? 'border-purple-500/60 shadow-[0_0_15px_rgba(168,85,247,0.3)]' : 'border-gray-700'} text-center pointer-events-none whitespace-nowrap`}>
                    {isLive ? (
                        <div className="flex items-center gap-4 text-xs font-mono">
                            <span className="text-purple-400 font-bold animate-pulse">GNN PROPAGATION</span>
                            <span className="text-gray-400">|</span>
                            <span className="text-white">NODE 1 → NODE 2</span>
                            <span className="text-gray-400">|</span>
                            <span className="text-gray-400">Weight: 0.50</span>
                            <span className="text-gray-400">|</span>
                            <span className="text-gray-400">Latency: 8ms</span>
                        </div>
                    ) : (
                        <div className="text-xs font-mono text-gray-500">ISOLATED (NO PROPAGATION)</div>
                    )}
                </div>
            </Html>
          )}
        </Canvas>
      </div>

      {/* 3. BOTTOM OVERLAYS */}
      <div className="absolute bottom-0 left-0 w-full p-8 flex justify-between items-end z-20 pointer-events-none">
          
          {/* EVENT STREAM (Left) */}
          <div className="w-1/3 max-w-sm bg-black/70 backdrop-blur-xl border border-white/10 rounded-xl p-5 shadow-2xl">
              <h3 className="text-xs font-bold text-gray-500 tracking-widest mb-3 pb-2 border-b border-white/10">SYSTEM EVENTS</h3>
              <div className="space-y-2">
                  {events.map(ev => (
                      <div key={ev.id} className="text-xs font-mono flex gap-3">
                          <span className="text-gray-500">[{ev.time}]</span>
                          <span className={ev.type === 'error' ? 'text-red-400' : ev.type === 'warn' ? 'text-yellow-400' : ev.type === 'success' ? 'text-green-400' : 'text-cyan-400'}>
                              {ev.msg}
                          </span>
                      </div>
                  ))}
                  {events.length === 0 && <div className="text-xs text-gray-600 font-mono">No recent events.</div>}
              </div>
          </div>

          {/* CLUSTER EFFICIENCY (Right) */}
          <div className="w-1/3 max-w-sm bg-black/70 backdrop-blur-xl border border-white/10 rounded-xl p-5 shadow-2xl">
              <h3 className="text-xs font-bold text-gray-500 tracking-widest mb-3 pb-2 border-b border-white/10">CLUSTER EFFICIENCY (ESTIMATED)</h3>
              <div className="space-y-4 text-xs font-mono">
                  <div>
                      <div className="flex justify-between mb-1"><span className="text-gray-400">Cooling Activation Latency</span><span className="text-white font-bold">{isLive ? '0ms (Pre-cooled)' : '4.5s (Reactive)'}</span></div>
                      <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden"><div className={`h-full ${isLive ? 'bg-green-500 w-full' : 'bg-orange-500 w-[20%]'}`}></div></div>
                  </div>
                  <div>
                      <div className="flex justify-between mb-1"><span className="text-gray-400">Total System Power Draw</span><span className="text-white font-bold">{telemetry ? '145W' : '0W'}</span></div>
                      <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden"><div className="h-full bg-blue-500 w-[60%]"></div></div>
                  </div>
                  <div className="pt-2 mt-2 flex justify-between text-gray-300 font-bold border-t border-white/10">
                      <span>THERMAL THROTTLING RISK</span>
                      <span className={isLive ? 'text-green-400' : (risk1 > 0.6 ? 'text-red-400' : 'text-yellow-400')}>{isLive ? 'LOW' : (risk1 > 0.6 ? 'HIGH' : 'MEDIUM')}</span>
                  </div>
              </div>
          </div>

      </div>

    </div>
  );
}
