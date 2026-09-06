import { useEffect } from 'react';

import { useTelemetry } from '../../services/telemetryApi';
import * as THREE from 'three';

const COLORS = {
  cool: new THREE.Color('#6F9BA8'),
  amber: new THREE.Color('#B58A4A'),
  orange: new THREE.Color('#C56A38'),
  critical: new THREE.Color('#B84A43'),
};

export function ThermalLayer({ scene }: { scene: THREE.Object3D }) {
  const { data: telemetry } = useTelemetry();

  useEffect(() => {
    if (!telemetry || !scene) return;

    scene.traverse((child) => {
      if (child.name === 'rack' && child.userData.rackId) {
        const rackData = telemetry.racks?.find((r) => r.id === child.userData.rackId);
        
        let targetColor = COLORS.cool;
        if (rackData) {
          if (rackData.risk_score >= 0.8) targetColor = COLORS.critical;
          else if (rackData.risk_score >= 0.6) targetColor = COLORS.orange;
          else if (rackData.risk_score >= 0.4) targetColor = COLORS.amber;
        }

        child.traverse((mesh: any) => {
          if (mesh.isMesh && mesh.material) {
            // Clone material once per mesh to allow independent colors
            if (!mesh.userData.uniqueMaterial) {
              mesh.material = mesh.material.clone();
              mesh.userData.uniqueMaterial = true;
            }
            mesh.material.emissive = targetColor;
            mesh.material.emissiveIntensity = rackData ? 0.5 : 0;
          }
        });
      }
    });
  }, [telemetry, scene]);

  return null;
}
