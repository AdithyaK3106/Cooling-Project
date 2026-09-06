import { useMemo } from 'react';
import { useTelemetry } from '../../services/telemetryApi';
import { Text, Billboard } from '@react-three/drei';
import * as THREE from 'three';

export function StatsLayer({ scene }: { scene: THREE.Object3D }) {
  const { data: telemetry } = useTelemetry();

  const rackPositions = useMemo(() => {
    const map = new Map<string, THREE.Vector3>();
    if (!scene) return map;
    scene.traverse((child) => {
      if (child.userData && child.userData.rackId) {
        const pos = new THREE.Vector3();
        child.getWorldPosition(pos);
        map.set(child.userData.rackId, pos);
      }
    });
    return map;
  }, [scene]);

  if (!telemetry || !rackPositions.size) return null;

  return (
    <>
      {telemetry.racks.map((rack) => {
        const pos = rackPositions.get(rack.id);
        if (!pos) return null;

        // Color temperature text based on risk
        let tempColor = "#00ffff"; // cool
        if (rack.risk_score >= 0.8) tempColor = "#ff3333";
        else if (rack.risk_score >= 0.6) tempColor = "#ff9900";
        else if (rack.risk_score >= 0.4) tempColor = "#ffff00";

        return (
          <Billboard key={rack.id} position={[pos.x, pos.y + 1.2, pos.z]} follow={true}>
            <Text
              position={[0, 0.15, 0]}
              fontSize={0.15}
              color="white"
              anchorX="center"
              anchorY="bottom"
              outlineWidth={0.015}
              outlineColor="black"
            >
              {rack.id}
            </Text>
            <Text
              position={[0, 0, 0]}
              fontSize={0.12}
              color={tempColor}
              anchorX="center"
              anchorY="bottom"
              outlineWidth={0.012}
              outlineColor="black"
            >
              {rack.telemetry.cpu_temp.toFixed(1)}°C
            </Text>
          </Billboard>
        );
      })}
    </>
  );
}
