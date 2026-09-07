import { useEffect, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { useTelemetry } from '../../services/telemetryApi';
import { useUiStore } from '../../stores/uiStore';
import * as THREE from 'three';

// Risk color targets — intentionally restrained/professional
const RISK_COLORS = {
  low:    new THREE.Color('#0ea5e9'), // sky-500 — cool blue
  medium: new THREE.Color('#f59e0b'), // amber-400
  high:   new THREE.Color('#ef4444'), // red-500
};

function getRiskColor(riskScore: number): THREE.Color {
  if (riskScore > 0.7) return RISK_COLORS.high;
  if (riskScore > 0.4) return RISK_COLORS.medium;
  return RISK_COLORS.low;
}

function getRiskEmissiveIntensity(riskScore: number): number {
  // Scale 0.05 (low) to 0.40 (critical) — subtle, enterprise-grade
  return 0.05 + riskScore * 0.35;
}

export function RiskColorLayer({ scene }: { scene: THREE.Object3D }) {
  const { data: telemetry } = useTelemetry();
  const { activeLayer } = useUiStore();
  const cachedRef = useRef(false);

  // One-time traversal: clone materials and build rack->mesh map
  useEffect(() => {
    if (!scene || cachedRef.current) return;

    const rackMap = new Map<string, any[]>();

    scene.traverse((child: any) => {
      if (child.userData && child.userData.rackId) {
        const meshes: any[] = [];
        child.traverse((mesh: any) => {
          if (mesh.isMesh && mesh.material) {
            // Clone material so we do not mutate shared GLTF materials
            if (!mesh.userData.__riskMaterialCloned) {
              mesh.material = Array.isArray(mesh.material)
                ? (mesh.material as any[]).map((m: any) => m.clone())
                : mesh.material.clone();
              mesh.userData.__riskMaterialCloned = true;
            }
            meshes.push(mesh);
          }
        });
        rackMap.set(child.userData.rackId, meshes);
      }
    });

    (scene as any).__riskColorRackMap = rackMap;
    cachedRef.current = true;
  }, [scene]);

  // Every frame: smoothly lerp each rack emissive toward its current risk color.
  // Skip when THERMAL mode is active — ThermalLayer owns emissive colors in that mode.
  useFrame((_state, delta) => {
    if (activeLayer === 'THERMAL') return;
    if (!telemetry?.racks || !cachedRef.current) return;

    const rackMap: Map<string, any[]> = (scene as any).__riskColorRackMap;
    if (!rackMap) return;

    // Smooth lerp factor — ~0.8s to fully settle on a color change
    const lerpFactor = Math.min(1, delta * 1.5);

    rackMap.forEach((meshes, rackId) => {
      const rackData = telemetry.racks?.find((r: any) => r.id === rackId);
      const riskScore = rackData?.risk_score ?? 0;
      const targetColor = getRiskColor(riskScore);
      const targetIntensity = getRiskEmissiveIntensity(riskScore);

      meshes.forEach((mesh: any) => {
        const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        mats.forEach((mat: any) => {
          if (mat.emissive !== undefined) {
            mat.emissive.lerp(targetColor, lerpFactor);
            mat.emissiveIntensity += (targetIntensity - mat.emissiveIntensity) * lerpFactor;
            mat.needsUpdate = true;
          }
        });
      });
    });
  });

  return null;
}
