import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';
import { Suspense } from 'react';
import { DataCenterScene } from '../three/DataCenterScene';
import { CameraController } from '../three/Camera';
import { SceneControlBar } from '../features/SceneControlBar';
import { ContextualDetailsPanel } from '../features/ContextualDetailsPanel';

export function ThermalMapTab() {
  return (
    <div className="relative h-full w-full bg-[#0B0E14] overflow-hidden">
      
      {/* 3D SCENE LAYER */}
      <div className="absolute inset-0 z-0">
        <Canvas shadows gl={{ antialias: true, alpha: false }}>
          <color attach="background" args={['#0B0E14']} />
          <ambientLight intensity={0.4} />
          <directionalLight position={[10, 20, 10]} intensity={1.5} castShadow />
          
          <Suspense fallback={null}>
            <DataCenterScene />
            <Environment preset="city" />
            <CameraController />
          </Suspense>

          <OrbitControls 
            makeDefault
            maxPolarAngle={Math.PI / 2 - 0.05} 
            minDistance={10} 
            maxDistance={200}
            enableDamping
            dampingFactor={0.05}
          />
        </Canvas>
      </div>

      {/* OVERLAY */}
      <div className="pointer-events-none absolute inset-0 z-10 flex flex-col justify-between p-6">
        <div className="flex justify-center">
          <SceneControlBar />
        </div>
        <div className="flex justify-end mt-6 flex-1">
          <div className="pointer-events-auto">
            <ContextualDetailsPanel />
          </div>
        </div>
      </div>
      
    </div>
  );
}
