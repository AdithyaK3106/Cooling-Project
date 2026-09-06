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
      // Ensure the rack hasn't been stripped from the scene
      if (child.userData && child.userData.rackId && child.parent !== null) {
        // Double check it's actually in the active scene graph
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
        const rack = telemetry.racks.find(r => r.id === rackId);

        if (!rack) {
          return (
            <Billboard key={rackId} position={[pos.x, pos.y + 1.2, pos.z]} follow={true}>
              <Text
                position={[0, 0, 0]}
                fontSize={0.08}
                color="#888888"
                anchorX="center"
                anchorY="bottom"
                outlineWidth={0.008}
                outlineColor="black"
              >
                WAITING FOR TELEMETRY
              </Text>
            </Billboard>
          );
        }

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
