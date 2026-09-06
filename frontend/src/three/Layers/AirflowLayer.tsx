import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// Generates simple moving particles/arrows in the aisles between racks
export function AirflowLayer() {
  const particlesCount = 200;
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  // Pre-generate particle data
  const particles = useMemo(() => {
    const temp = [];
    for (let i = 0; i < particlesCount; i++) {
      // Spawn particles randomly in the aisle areas
      // Racks are at Z = -10, -5, 0, 5, 10
      // Aisles are roughly between them
      
      const x = (Math.random() - 0.5) * 20; // Spread across X
      
      // Pick a random aisle (between Z rows)
      const aisleIndex = Math.floor(Math.random() * 4);
      const aisles = [-7.5, -2.5, 2.5, 7.5];
      const z = aisles[aisleIndex] + (Math.random() - 0.5) * 1.5;
      
      const y = Math.random() * 8 + 0.5; // Height

      temp.push({
        position: new THREE.Vector3(x, y, z),
        speed: Math.random() * 0.05 + 0.02,
        phase: Math.random() * Math.PI * 2
      });
    }
    return temp;
  }, []);

  useFrame(() => {
    if (!meshRef.current) return;

    particles.forEach((p, i) => {
      // Move particles along X axis to simulate airflow
      p.position.x += p.speed;
      
      // Gently bob up and down
      p.position.y += Math.sin(p.phase + performance.now() * 0.002) * 0.01;

      // Wrap around
      if (p.position.x > 10) {
        p.position.x = -10;
      }

      dummy.position.copy(p.position);
      
      // Stretch particles along movement vector
      dummy.scale.set(1.5, 0.2, 0.2);
      dummy.updateMatrix();
      
      meshRef.current!.setMatrixAt(i, dummy.matrix);
    });

    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, particlesCount]}>
      <boxGeometry args={[1, 1, 1]} />
      <meshBasicMaterial color="#06b6d4" transparent opacity={0.3} blending={THREE.AdditiveBlending} depthWrite={false} />
    </instancedMesh>
  );
}
