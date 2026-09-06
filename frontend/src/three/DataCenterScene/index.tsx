import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, useGLTF } from '@react-three/drei';
import { CameraController } from '../Camera';

import { useUiStore } from '../../stores/uiStore';

import { ThermalLayer } from '../Layers/ThermalLayer';
import { CoolingLayer } from '../Layers/CoolingLayer';
import { GNNLayer } from '../Layers/GNNLayer';

function Model() {
  const { scene } = useGLTF('/models/room_server.glb');
  const setSelectedRackId = useUiStore((state) => state.setSelectedRackId);

  // Traverse the scene once to assign IDs and attach event handlers
  React.useEffect(() => {
    let rackCounter = 1;
    scene.traverse((child) => {
      if (child.name === 'rack') {
        const rackId = `A0${rackCounter++}`;
        child.userData = { rackId };
      }
    });
  }, [scene]);

  return (
    <group>
      <primitive 
        object={scene} 
        onClick={(e: any) => {
          e.stopPropagation();
          let node = e.object;
          while (node && node.name !== 'rack' && node.parent) {
            node = node.parent;
          }
          if (node && node.name === 'rack') {
            setSelectedRackId(node.userData.rackId);
          } else {
            setSelectedRackId(null);
          }
        }}
        onPointerMissed={() => setSelectedRackId(null)}
      />
      
      <ThermalLayer scene={scene} />
      <CoolingLayer scene={scene} />
      <GNNLayer scene={scene} />
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
