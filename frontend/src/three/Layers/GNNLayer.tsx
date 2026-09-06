import { useMemo } from 'react';
import { useTelemetry } from '../../services/telemetryApi';
import * as THREE from 'three';
import { Line } from '@react-three/drei';

export function GNNLayer({ scene }: { scene: THREE.Object3D }) {
  const { data: telemetry } = useTelemetry();

  const edges = useMemo(() => {
    if (!telemetry || !scene || !telemetry.topology) return [];
    
    // Build a map of rackId to world position
    const positions: Record<string, THREE.Vector3> = {};
    scene.traverse((child) => {
      if (child.name === 'rack' && child.userData.rackId) {
        const pos = new THREE.Vector3();
        child.getWorldPosition(pos);
        positions[child.userData.rackId] = pos;
      }
    });

    return telemetry.topology
      .filter((edge) => positions[edge.source] && positions[edge.target])
      .map((edge) => {
        const start = positions[edge.source];
        const end = positions[edge.target];
        
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
  }, [telemetry, scene]);

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
