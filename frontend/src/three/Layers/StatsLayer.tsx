import { useMemo } from 'react';
import { useTelemetry } from '../../services/telemetryApi';
import { Html } from '@react-three/drei';
import * as THREE from 'three';

export function StatsLayer({ scene }: { scene: THREE.Object3D }) {
  const { data: telemetry } = useTelemetry();

  const rackPositions = useMemo(() => {
    const map = new Map<string, THREE.Vector3>();
    if (!scene) return map;
    scene.traverse((child) => {
      if (child.userData && child.userData.rackId && child.parent !== null) {
        let isActive = true;
        let node = child;
        while (node) {
          if (!node.parent && node !== scene) isActive = false;
          node = node.parent as any;
        }

        if (isActive) {
          const pos = new THREE.Vector3();
          child.getWorldPosition(pos);
          map.set(child.userData.rackId, pos);
        }
      }
    });
    return map;
  }, [scene]);

  if (!telemetry || !rackPositions.size) return null;

  return (
    <>
      {Array.from(rackPositions.entries()).map(([rackId, pos]) => {
        const rack = telemetry.racks.find((r: any) => r.id === rackId);
        if (!rack) return null;

        const risk = rack.risk_score || 0;
        const riskPct = Math.round(risk * 100);

        let badgeBg = 'bg-emerald-950/80 border-emerald-500/80 text-emerald-300';
        if (risk >= 0.75) badgeBg = 'bg-red-950/90 border-red-500 text-red-200 animate-pulse';
        else if (risk >= 0.55) badgeBg = 'bg-orange-950/80 border-orange-500 text-orange-200';
        else if (risk >= 0.35) badgeBg = 'bg-amber-950/80 border-amber-500 text-amber-200';

        const isCooled = rack.cooling?.status === 'predictive intervention' || rack.cooling?.override;

        return (
          <Html
            key={rack.id}
            position={[pos.x, pos.y + 2.5, pos.z]}
            center
            style={{ pointerEvents: 'none', userSelect: 'none' }}
          >
            <div className={`px-2 py-1 rounded-lg border backdrop-blur-md shadow-lg flex flex-col items-center gap-0.5 text-[10px] font-mono whitespace-nowrap transition-all duration-300 ${badgeBg}`}>
              <div className="font-bold tracking-wider flex items-center gap-1 text-white">
                <span>{rack.id}</span>
                {isCooled && <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />}
              </div>
              <div className="flex items-center gap-1.5 text-[9px]">
                <span className="font-bold">RISK {riskPct}%</span>
                <span className="opacity-70">CPU {rack.telemetry?.cpu_util?.toFixed(0)}%</span>
              </div>
            </div>
          </Html>
        );
      })}
    </>
  );
}
