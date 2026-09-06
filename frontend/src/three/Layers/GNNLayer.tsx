import { useMemo } from 'react';
import { useTelemetry } from '../../services/telemetryApi';
import * as THREE from 'three';
import { Line } from '@react-three/drei';

export function GNNLayer({ scene }: { scene: THREE.Object3D }) {
  const { data: telemetry } = useTelemetry();

  const positions = useMemo(() => {
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

  const edges = useMemo(() => {
    if (!telemetry || !telemetry.topology) return [];

    return telemetry.topology
      .filter((edge) => positions.has(edge.source) && positions.has(edge.target))
      .map((edge) => {
        const start = positions.get(edge.source)!;
        const end = positions.get(edge.target)!;
        
        // Create a bezier curve between the nodes
        const mid = start.clone().lerp(end, 0.5);
        mid.y += 5 * edge.weight; // Arch upwards based on weight

        const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
        const points = curve.getPoints(20);
        
        return {
          points,
          weight: edge.weight,
        };
      });
  }, [telemetry, positions]);

  if (telemetry?.operating_mode !== 'DATA_CENTER_SIMULATION') return null;

  return (
    <>
      {edges.map((edge, i) => (
        <Line 
          key={i} 
          points={edge.points} 
          color="#C56A38" 
          lineWidth={edge.weight * 2} 
          transparent 
          opacity={0.6} 
        />
      ))}
    </>
  );
}
