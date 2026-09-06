import { DataCenterScene } from '../three/DataCenterScene';
import { SceneControlBar } from '../features/SceneControlBar';
import { ContextualDetailsPanel } from '../features/ContextualDetailsPanel';

export function ThermalMapTab() {
  return (
    <div className="relative h-full w-full bg-[#0B0E14] overflow-hidden">
      
      {/* 3D SCENE LAYER */}
      <div className="absolute inset-0 z-0">
        <DataCenterScene />
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
