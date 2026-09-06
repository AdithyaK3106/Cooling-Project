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
    
    // Direct iteration over the cached map (O(N) where N=25 racks) instead of full scene traversal
    rackMap.forEach((meshes, rackId) => {
      const rackData = telemetry.racks?.find((r) => r.id === rackId);
      
      let targetColor = COLORS.cool;
      if (rackData) {
        if (rackData.risk_score >= 0.8) targetColor = COLORS.critical;
        else if (rackData.risk_score >= 0.6) targetColor = COLORS.orange;
        else if (rackData.risk_score >= 0.4) targetColor = COLORS.amber;
      }
      
      const intensity = rackData ? 0.2 + rackData.risk_score * 1.5 : 0.0;

      meshes.forEach((mesh) => {
        const activeMats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        activeMats.forEach((mat: any) => {
          if (mat.emissive !== undefined) {
            mat.emissive.copy(targetColor);
            mat.emissiveIntensity = intensity;
          }
          mat.needsUpdate = true;
        });
      });
    });
  }, [telemetry, scene]);

  return null;
}
