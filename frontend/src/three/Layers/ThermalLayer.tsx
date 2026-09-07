import { useEffect } from 'react';

import { useTelemetry } from '../../services/telemetryApi';
import * as THREE from 'three';

const COLORS = {
  emerald: new THREE.Color('#10b981'), // Low (<35%)
  amber: new THREE.Color('#f59e0b'),   // Medium (35-54%)
  orange: new THREE.Color('#f97316'),  // High (55-74%)
  critical: new THREE.Color('#ef4444'),// Critical (>=75%)
};

export function ThermalLayer({ scene }: { scene: THREE.Object3D }) {
  const { data: telemetry } = useTelemetry();

  useEffect(() => {
    if (!scene) return;
    
    // Cache meshes and initialize materials ONCE
    if (!(scene as any).__thermalCached) {
      const rackMap = new Map<string, any[]>();
      
      scene.traverse((child) => {
        if (child.userData && child.userData.rackId) {
          const meshes: any[] = [];
          
          child.traverse((mesh: any) => {
            if (mesh.isMesh && mesh.material) {
              const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
              
              if (!mesh.userData.uniqueMaterial) {
                mesh.material = Array.isArray(mesh.material) 
                  ? materials.map((m: any) => m.clone())
                  : mesh.material.clone();
                mesh.userData.uniqueMaterial = true;
                
                // Only add outlines to the main rack body to save draw calls
                if (mesh.name.match(/Rack body/i)) {
                  const edges = new THREE.EdgesGeometry(mesh.geometry);
                  const lineMat = new THREE.LineBasicMaterial({ color: 0x000000, linewidth: 2 });
                  const line = new THREE.LineSegments(edges, lineMat);
                  mesh.add(line);
                }
              }
              
              meshes.push(mesh);
            }
          });
          
          rackMap.set(child.userData.rackId, meshes);
        }
      });
      
      (scene as any).__thermalRackMap = rackMap;
      (scene as any).__thermalCached = true;
    }
  }, [scene]);

  useEffect(() => {
    if (!telemetry || !scene || !(scene as any).__thermalCached) return;

    const rackMap: Map<string, any[]> = (scene as any).__thermalRackMap;
    
    rackMap.forEach((meshes, rackId) => {
      const rackData = telemetry.racks?.find((r: any) => r.id === rackId);
      
      let targetColor = COLORS.emerald;
      if (rackData) {
        const risk = rackData.risk_score || 0;
        if (risk >= 0.75) targetColor = COLORS.critical;
        else if (risk >= 0.55) targetColor = COLORS.orange;
        else if (risk >= 0.35) targetColor = COLORS.amber;
        else targetColor = COLORS.emerald;
      }
      
      const intensity = rackData ? (0.3 + rackData.risk_score * 1.6) : 0.2;

      meshes.forEach((mesh) => {
        const activeMats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        activeMats.forEach((mat: any) => {
          if (mat.emissive !== undefined) {
            mat.emissive.copy(targetColor);
            mat.emissiveIntensity = intensity;
          }
          if (mat.color !== undefined && rackData) {
            mat.color.lerp(targetColor, 0.4);
          }
          mat.needsUpdate = true;
        });
      });
    });
  }, [telemetry, scene]);

  return null;
}
