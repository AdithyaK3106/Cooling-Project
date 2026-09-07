import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, useGLTF } from '@react-three/drei';
import { CameraController } from '../Camera';

import { useUiStore } from '../../stores/uiStore';

import { ThermalLayer } from '../Layers/ThermalLayer';
import { GNNLayer } from '../Layers/GNNLayer';
import { AirflowLayer } from '../Layers/AirflowLayer';
import { StatsLayer } from '../Layers/StatsLayer';
import { Html } from '@react-three/drei';
import { useTelemetry } from '../../services/telemetryApi';
import * as THREE from 'three';

function Model() {
  const { scene } = useGLTF('/models/room_server.glb');
  const { setSelectedRackId, setHoveredRackId, activeLayer } = useUiStore();
  const hoveredRackId = useUiStore((state) => state.hoveredRackId);
  const { data: telemetry } = useTelemetry();

  const [isReady, setIsReady] = React.useState(false);

  // Traverse the scene once to assign IDs, attach event handlers, and STRIP excess racks
  React.useEffect(() => {
    const nodesToRemove: THREE.Object3D[] = [];

    scene.traverse((child: any) => {
      // Hide roof/ceiling so we can see inside and click
      if (child.name.toLowerCase().includes('roof') || child.name.toLowerCase().includes('ceiling') || child.name.toLowerCase().includes('top plane')) {
        child.visible = false;
      }

      // Check if this object or any ancestor is a Rack
      let isPartOfRack = false;
      let node = child;
      while (node) {
        if (node.name.match(/Rack/i)) {
          isPartOfRack = true;
          break;
        }
        node = node.parent;
      }

      // Disable raycasting on non-rack objects (walls, floors) so hover works!
      if (!isPartOfRack) {
        child.raycast = () => null;
      }

      const match = child.name.match(/^Rack\s*(\d+)$/i) || child.name.match(/^Rack_(\d+)$/i);
      if (match) {
        const num = parseInt(match[1], 10);
        
        // Strip out everything except the first 25 racks
        if (num > 25) {
          nodesToRemove.push(child);
          return;
        }

        const rackId = `A0${num}`;
        child.userData = { rackId };

        // Force into a perfect 5x5 grid centered at origin
        const index = num - 1;
        const row = Math.floor(index / 5);
        const col = index % 5;
        
        // Adjust spacing based on visual preference
        const spacingX = 3.5;
        const spacingZ = 5.0;
        
        child.position.set(
          (col - 2) * spacingX,
          0,
          (row - 2) * spacingZ
        );
        // Reset rotation so they all face forward neatly
        child.rotation.set(0, 0, 0);
      }
    });

    // Remove the excess racks from the scene graph entirely
    nodesToRemove.forEach(node => node.removeFromParent());
    
    setIsReady(true);
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
    return telemetry?.racks.find((r: any) => r.id === hoveredRackId);
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
          <div className="bg-[#0B0E14]/90 border border-white/10 p-3 rounded-lg shadow-2xl text-xs w-48 text-gray-200 backdrop-blur-xl">
            <div className="font-mono font-bold text-white mb-2 border-b border-white/10 pb-1.5 flex justify-between items-center">
              <span>{hoveredRackId}</span>
              <span className={`w-1.5 h-1.5 rounded-full ${hoveredRackData.risk_score > 0.7 ? 'bg-red-500' : 'bg-green-500'}`}></span>
            </div>
            <div className="flex justify-between py-0.5">
              <span className="text-gray-400 font-sans">CPU</span>
              <span className="font-mono text-cyan-400">{hoveredRackData.telemetry.cpu_util.toFixed(1)}%</span>
            </div>
            <div className="flex justify-between py-0.5">
              <span className="text-gray-400 font-sans">Temp</span>
              <span className="font-mono text-orange-400">{hoveredRackData.telemetry.cpu_temp.toFixed(1)}°C</span>
            </div>
          </div>
        </Html>
      )}

      {isReady && (
        <>
          <StatsLayer scene={scene} />
          {activeLayer === 'THERMAL' && <ThermalLayer scene={scene} />}
          {activeLayer === 'RISK' && <GNNLayer scene={scene} />}
          {activeLayer === 'AIRFLOW' && <AirflowLayer />}
        </>
      )}
    </group>
  );
}

import { useThree } from '@react-three/fiber';

function PerfExposer() {
  const { gl, scene } = useThree();
  React.useEffect(() => {
    (window as any).__gl = gl;
    (window as any).__scene = scene;
  }, [gl, scene]);
  return null;
}

export function DataCenterScene() {
  return (
    <div className="h-full w-full">
      <Canvas camera={{ position: [50, 50, 50], fov: 45 }}>
        <PerfExposer />
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
