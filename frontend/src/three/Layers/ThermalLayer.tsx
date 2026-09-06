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
      if (child.userData && child.userData.rackId) {
        const rackData = telemetry.racks?.find((r) => r.id === child.userData.rackId);
        
        let targetColor = COLORS.cool;
        if (rackData) {
          if (rackData.risk_score >= 0.8) targetColor = COLORS.critical;
          else if (rackData.risk_score >= 0.6) targetColor = COLORS.orange;
          else if (rackData.risk_score >= 0.4) targetColor = COLORS.amber;
        }

        console.log(`ThermalLayer checking rack ${child.userData.rackId}, data exists: ${!!rackData}`);

        console.log(`ThermalLayer: Rack ${child.userData.rackId} matched. Data exists: ${!!rackData}, Color:`, targetColor);

        child.traverse((mesh: any) => {
          if (mesh.isMesh && mesh.material) {
            // Handle arrays of materials
            const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
            
            if (!mesh.userData.uniqueMaterial) {
              mesh.material = Array.isArray(mesh.material) 
                ? materials.map((m: any) => m.clone())
                : mesh.material.clone();
              mesh.userData.uniqueMaterial = true;
              
              // Add black outlines as requested by user
              const edges = new THREE.EdgesGeometry(mesh.geometry);
              const lineMat = new THREE.LineBasicMaterial({ color: 0x000000, linewidth: 2 });
              const line = new THREE.LineSegments(edges, lineMat);
              mesh.add(line);
            }

            const activeMats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
            
            activeMats.forEach((mat: any) => {
              if (mat.emissive !== undefined) {
                mat.emissive.copy(targetColor);
                mat.emissiveIntensity = rackData ? 0.2 + rackData.risk_score * 1.5 : 0.0;
              }
              mat.needsUpdate = true;
            });
          }
        });
      }
    });
  }, [telemetry, scene]);

  return null;
}
