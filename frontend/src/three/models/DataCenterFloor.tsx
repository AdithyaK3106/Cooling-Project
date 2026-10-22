import { useMemo } from 'react';
import * as THREE from 'three';

/**
 * High-tech enterprise data center raised floor system.
 * Matches the reference image with:
 * - Precision-fitted interior raised modular floor tiles with grid seams
 * - High-density perforated metal ventilation floor grates along server aisles
 * - Subtle underfloor cold-plenum glow
 * - Dark graphite exterior perimeter floor border
 */
export function DataCenterFloor() {
  const { floorMaterial, ventMaterial, exteriorMaterial, interiorGeom, exteriorGeom, ventGeom, ventPositions } = useMemo(() => {
    // 1. Procedural tile texture with subtle surface grain and grid joint seams
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 1024;
    const ctx = canvas.getContext('2d')!;

    // Base slate/concrete tile color matching reference image
    ctx.fillStyle = '#8e9eaf';
    ctx.fillRect(0, 0, 1024, 1024);

    // Micro-texture noise for realistic anti-static laminate
    const imgData = ctx.getImageData(0, 0, 1024, 1024);
    const data = imgData.data;
    for (let i = 0; i < data.length; i += 4) {
      const noise = (Math.random() - 0.5) * 12;
      data[i] = Math.min(255, Math.max(0, data[i] + noise));
      data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + noise));
      data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + noise));
    }
    ctx.putImageData(imgData, 0, 0);

    // Draw square tile grid lines (600mm raised floor tiles)
    const numTiles = 16;
    const tileSize = 1024 / numTiles;
    ctx.strokeStyle = '#384252';
    ctx.lineWidth = 3.0;

    for (let x = 0; x <= numTiles; x++) {
      ctx.beginPath();
      ctx.moveTo(x * tileSize, 0);
      ctx.lineTo(x * tileSize, 1024);
      ctx.stroke();
    }
    for (let y = 0; y <= numTiles; y++) {
      ctx.beginPath();
      ctx.moveTo(0, y * tileSize);
      ctx.lineTo(1024, y * tileSize);
      ctx.stroke();
    }

    const tileTexture = new THREE.CanvasTexture(canvas);
    tileTexture.wrapS = THREE.RepeatWrapping;
    tileTexture.wrapT = THREE.RepeatWrapping;
    tileTexture.repeat.set(2.5, 3.2);

    // 2. High-contrast perforated airflow ventilation floor grate texture
    const ventCanvas = document.createElement('canvas');
    ventCanvas.width = 256;
    ventCanvas.height = 256;
    const vCtx = ventCanvas.getContext('2d')!;

    // Steel grate frame
    vCtx.fillStyle = '#526071';
    vCtx.fillRect(0, 0, 256, 256);

    // Dark recessed honeycomb / perforation holes
    vCtx.fillStyle = '#0f172a';
    const holeRadius = 3.5;
    const spacing = 10;
    for (let vx = spacing / 2; vx < 256; vx += spacing) {
      for (let vy = spacing / 2; vy < 256; vy += spacing) {
        vCtx.beginPath();
        vCtx.arc(vx, vy, holeRadius, 0, Math.PI * 2);
        vCtx.fill();
      }
    }
    // High-contrast outer beveled rim
    vCtx.strokeStyle = '#222d3d';
    vCtx.lineWidth = 10;
    vCtx.strokeRect(5, 5, 246, 246);

    const ventTexture = new THREE.CanvasTexture(ventCanvas);
    ventTexture.wrapS = THREE.RepeatWrapping;
    ventTexture.wrapT = THREE.RepeatWrapping;

    // Interior floor PBR material
    const floorMaterial = new THREE.MeshStandardMaterial({
      map: tileTexture,
      roughness: 0.36,
      metalness: 0.22,
    });

    // Perforated vent tile material with subtle cold-aisle underfloor glow
    const ventMaterial = new THREE.MeshStandardMaterial({
      map: ventTexture,
      roughness: 0.30,
      metalness: 0.55,
      emissive: new THREE.Color('#0284c7'),
      emissiveIntensity: 0.15,
    });

    // Dark exterior floor border (matching dark obsidian environment in reference)
    const exteriorMaterial = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#0a0c11'),
      roughness: 0.90,
      metalness: 0.10,
    });

    // Room interior dimensions: span inside walls (X: ~19m, Z: ~25.8m)
    const interiorGeom = new THREE.PlaneGeometry(18.9, 25.7);
    // Tight exterior perimeter border slab
    const exteriorGeom = new THREE.PlaneGeometry(21.2, 28.0);

    // Perforated vent tile positions along cold aisles in front of racks
    // Racks are at X: [-7, -3.5, 0, 3.5, 7], Z: [-10, -5, 0, 5, 10]
    // Aisle centers between rack rows: Z = -7.5, -2.5, 2.5, 7.5
    const positions: [number, number][] = [];
    const aisleZs = [-7.5, -2.5, 2.5, 7.5];
    const rackXs = [-7, -3.5, 0, 3.5, 7];

    for (const z of aisleZs) {
      for (const x of rackXs) {
        positions.push([x, z]);
      }
    }

    const singleVentGeom = new THREE.PlaneGeometry(1.4, 1.4);

    return {
      floorMaterial,
      ventMaterial,
      exteriorMaterial,
      interiorGeom,
      exteriorGeom,
      ventGeom: singleVentGeom,
      ventPositions: positions,
    };
  }, []);

  return (
    <group name="DataCenterFloorSystem">
      {/* 1. Exterior Dark Border Slab */}
      <mesh
        geometry={exteriorGeom}
        material={exteriorMaterial}
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, -0.015, 0]}
        receiveShadow
        raycast={() => null}
      />

      {/* 2. Interior Precision Raised Floor */}
      <mesh
        geometry={interiorGeom}
        material={floorMaterial}
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0, 0]}
        receiveShadow
        raycast={() => null}
      />

      {/* 3. Perforated Airflow Ventilation Floor Grates (Cold Aisles) */}
      {ventPositions.map(([x, z], idx) => (
        <mesh
          key={`vent-${idx}`}
          geometry={ventGeom}
          material={ventMaterial}
          rotation={[-Math.PI / 2, 0, 0]}
          position={[x, 0.005, z]}
          receiveShadow
          raycast={() => null}
        />
      ))}
    </group>
  );
}
