import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, useGLTF } from '@react-three/drei';
import { CameraController } from '../Camera';

import { useUiStore } from '../../stores/uiStore';

import { ThermalLayer } from '../Layers/ThermalLayer';
import { GNNLayer } from '../Layers/GNNLayer';
import { StatsLayer } from '../Layers/StatsLayer';
import { Html } from '@react-three/drei';
import { useTelemetry } from '../../services/telemetryApi';
import * as THREE from 'three';

function Model() {
  const { scene } = useGLTF('/models/room_server.glb');
  const setSelectedRackId = useUiStore((state) => state.setSelectedRackId);
  const setHoveredRackId = useUiStore((state) => state.setHoveredRackId);
  const hoveredRackId = useUiStore((state) => state.hoveredRackId);
  const { data: telemetry } = useTelemetry();

  // Traverse the scene once to assign IDs and attach event handlers
  React.useEffect(() => {
    scene.traverse((child: any) => {
      // Hide roof/ceiling so we can see inside and click
      if (child.name.toLowerCase().includes('roof') || child.name.toLowerCase().includes('ceiling') || child.name.toLowerCase().includes('top plane')) {
        child.visible = false;
      }
      
      // Disable raycasting on non-rack objects (walls, floors) so hover works!
      if (!child.name.match(/Rack/i)) {
        child.raycast = () => null;
      }

      const match = child.name.match(/^Rack\s*(\d+)$/i) || child.name.match(/^Rack_(\d+)$/i);
      if (match) {
        const num = match[1];
        const rackId = `A0${num}`;
        child.userData = { rackId };
      }
    });
  }, [scene]);

  // Find the position of the hovered rack for the tooltip
  const hoveredRackPos = React.useMemo(() => {
    if (!hoveredRackId) return null;
    let pos = new THREE.Vector3();
    scene.traverse((child) => {
      if (child.userData.rackId === hoveredRackId) {
        child.getWorldPosition(pos);
        pos.y += 3; // Float above the rack
      }
    });
    return pos;
  }, [hoveredRackId, scene]);

  const hoveredRackData = React.useMemo(() => {
    return telemetry?.racks.find((r) => r.id === hoveredRackId);
  }, [telemetry, hoveredRackId]);

  return (
    <group>
      <primitive 
        object={scene} 
        onClick={(e: any) => {
          e.stopPropagation();
          let node = e.object;
          while (node && !(node.name.match(/^Rack\s*\d+$/i) || node.name.match(/^Rack_\d+$/i)) && node.parent) {
            node = node.parent;
          }
          if (node && (node.name.match(/^Rack\s*\d+$/i) || node.name.match(/^Rack_\d+$/i))) {
            setSelectedRackId(node.userData.rackId);
          } else {
            setSelectedRackId(null);
          }
        }}
        onPointerOver={(e: any) => {
          e.stopPropagation();
          let node = e.object;
          while (node && !(node.name.match(/^Rack\s*\d+$/i) || node.name.match(/^Rack_\d+$/i)) && node.parent) {
            node = node.parent;
          }
          if (node && (node.name.match(/^Rack\s*\d+$/i) || node.name.match(/^Rack_\d+$/i))) {
            setHoveredRackId(node.userData.rackId);
            document.body.style.cursor = 'pointer';
          }
        }}
        onPointerOut={(e: any) => {
          e.stopPropagation();
          setHoveredRackId(null);
          document.body.style.cursor = 'default';
        }}
        onPointerMissed={() => {
          setSelectedRackId(null);
          setHoveredRackId(null);
        }}
      />
      
      {hoveredRackPos && hoveredRackData && (
        <Html position={hoveredRackPos} center style={{ pointerEvents: 'none', zIndex: 100 }}>
          <div className="bg-[#1C1F26] border border-[#2D3342] p-3 rounded shadow-lg text-xs w-48 text-gray-200 backdrop-blur-md bg-opacity-90">
            <div className="font-bold text-white mb-1 border-b border-[#2D3342] pb-1">{hoveredRackId}</div>
            <div className="flex justify-between">
              <span className="text-gray-400">CPU</span>
              <span className="font-mono text-cyan-400">{hoveredRackData.telemetry.cpu_util.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Temp</span>
              <span className="font-mono text-orange-400">{hoveredRackData.telemetry.cpu_temp.toFixed(1)}°C</span>
            </div>
            <div className="flex justify-between mt-1 pt-1 border-t border-[#2D3342]">
              <span className="text-gray-400">Risk</span>
              <span className={`font-mono font-bold ${hoveredRackData.risk_score > 0.7 ? 'text-red-400' : 'text-green-400'}`}>
                {(hoveredRackData.risk_score * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </Html>
      )}

      <ThermalLayer scene={scene} />
      <GNNLayer scene={scene} />
      <StatsLayer scene={scene} />
    </group>
  );
}

export function DataCenterScene() {
  return (
    <div className="h-full w-full">
      <Canvas camera={{ position: [50, 50, 50], fov: 45 }}>
        <color attach="background" args={['#1A1C23']} />
        
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 20, 10]} intensity={1.5} />
        
        <Suspense fallback={null}>
          <Model />
          <Environment preset="city" />
        </Suspense>

        <OrbitControls makeDefault />
        <CameraController />
      </Canvas>
    </div>
  );
}
