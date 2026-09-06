import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, useGLTF } from '@react-three/drei';

function Model() {
  const { scene } = useGLTF('/room_server.gltf'); // We will adjust the path if it exports as .glb
  return <primitive object={scene} />;
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
      </Canvas>
    </div>
  );
}
