import { useMemo } from 'react';
import { useTelemetry } from '../../services/telemetryApi';
import * as THREE from 'three';

export function CoolingLayer({ scene }: { scene: THREE.Object3D }) {
  const { data: telemetry } = useTelemetry();

  const activeCoolingRacks = useMemo(() => {
    if (!telemetry || !scene) return [];
    
    const positions: THREE.Vector3[] = [];
    scene.traverse((child) => {
      if (child.userData && child.userData.rackId) {
        const rackData = telemetry.racks?.find((r) => r.id === child.userData.rackId);
        // "predictive intervention" active
        if (rackData && rackData.cooling.status === 'predictive intervention') {
          const pos = new THREE.Vector3();
          child.getWorldPosition(pos);
          positions.push(pos);
        }
      }
    });
    return positions;
  }, [telemetry, scene]);

  return (
    <>
      {activeCoolingRacks.map((pos, i) => (
        <mesh key={i} position={[pos.x, pos.y + 2, pos.z]}>
          <cylinderGeometry args={[0.5, 0.5, 4, 8]} />
          <meshBasicMaterial color="#6C8FA0" transparent opacity={0.4} wireframe />
        </mesh>
      ))}
    </>
  );
}
