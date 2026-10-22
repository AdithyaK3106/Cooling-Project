import { useMemo } from 'react';
import * as THREE from 'three';

/**
 * PerimeterWall: High-tech modular enterprise data-center enclosure.
 * Exactly matches reference image:
 * - Obsidian/graphite modular wall panels with architectural reveals
 * - Structural corner and intermediate pilasters with embedded vertical glowing LED light columns
 * - Front glass entrance framed by illuminated pillars and glass sliding panels
 * - Interior downward architectural wall wash lights creating atmospheric pools on the floor
 * - Disabled raycasting to preserve seamless hover and click transparency for server racks
 */
export function PerimeterWall() {
  const { materials, geometries } = useMemo(() => {
    // 1. Dark graphite metallic materials
    const wallBodyMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#141720'),
      metalness: 0.70,
      roughness: 0.38,
    });

    const wallTrimMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#0e1016'),
      metalness: 0.88,
      roughness: 0.28,
    });

    const capMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#1f2430'),
      metalness: 0.80,
      roughness: 0.25,
    });

    // Vertical LED light strip material (pure brilliant white emission)
    const ledStripMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#f8fafc'),
      emissive: new THREE.Color('#f8fafc'),
      emissiveIntensity: 2.2,
      roughness: 0.1,
    });

    // Wall sconce luminaire lens (cool white/cyan architectural wash)
    const sconceLensMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#e0f2fe'),
      emissive: new THREE.Color('#bae6fd'),
      emissiveIntensity: 2.0,
      roughness: 0.1,
    });

    const sconceHousingMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#0d0f14'),
      metalness: 0.92,
      roughness: 0.2,
    });

    // High-tech architectural glass door
    const glassDoorMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#1e293b'),
      metalness: 0.90,
      roughness: 0.08,
      transparent: true,
      opacity: 0.35,
      depthWrite: false,
    });

    const glassFrameMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#2c323f'),
      metalness: 0.88,
      roughness: 0.25,
    });

    // Enclosure dimensions: span enclosing 5x5 racks (±9.6m on X, ±13.0m on Z)
    const wallHeight = 3.20;
    const wallThick = 0.28;
    const plinthHeight = 0.18;
    const capHeight = 0.10;

    const sideLength = 26.0; // Z span (-13 to +13)
    const frontHalfLength = 7.4; // X span from corner ±9.6 to portal ±2.2

    return {
      materials: {
        wallBodyMat,
        wallTrimMat,
        capMat,
        ledStripMat,
        sconceLensMat,
        sconceHousingMat,
        glassDoorMat,
        glassFrameMat,
      },
      geometries: {
        // Continuous side walls
        sideWall: new THREE.BoxGeometry(wallThick, wallHeight - plinthHeight - capHeight, sideLength),
        sidePlinth: new THREE.BoxGeometry(wallThick + 0.05, plinthHeight, sideLength),
        sideCap: new THREE.BoxGeometry(wallThick + 0.08, capHeight, sideLength + 0.08),

        // Front & back half-walls with central entrance portal
        frontWallHalf: new THREE.BoxGeometry(frontHalfLength, wallHeight - plinthHeight - capHeight, wallThick),
        frontPlinthHalf: new THREE.BoxGeometry(frontHalfLength, plinthHeight, wallThick + 0.05),
        frontCapHalf: new THREE.BoxGeometry(frontHalfLength + 0.04, capHeight, wallThick + 0.08),

        // Portal lintel beam
        portalLintel: new THREE.BoxGeometry(4.4, 0.40, wallThick + 0.04),

        // Structural columns / pilasters
        pilaster: new THREE.BoxGeometry(0.44, wallHeight + 0.08, 0.44),

        // Vertical embedded LED light column strips (height 1.8m)
        verticalLedStrip: new THREE.BoxGeometry(0.05, 1.80, 0.03),

        // Glass door panels
        glassPanel: new THREE.BoxGeometry(1.85, 2.45, 0.04),
        glassHandle: new THREE.BoxGeometry(0.03, 1.10, 0.06),

        // Architectural wall wash sconce
        sconceHousing: new THREE.BoxGeometry(0.26, 0.08, 0.14),
        sconceLens: new THREE.BoxGeometry(0.22, 0.02, 0.10),
      },
    };
  }, []);

  const wallHeight = 3.20;
  const bodyY = 0.18 + (wallHeight - 0.18 - 0.10) / 2; // ~1.64m
  const capY = wallHeight - 0.05;                      // ~3.15m
  const plinthY = 0.09;                               // ~0.09m

  return (
    <group name="DataCenterPerimeterWall">
      {/* ─── 1. LEFT PERIMETER WALL (X = -9.6m) ─── */}
      <mesh geometry={geometries.sideWall} material={materials.wallBodyMat} position={[-9.6, bodyY, 0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.sidePlinth} material={materials.wallTrimMat} position={[-9.6, plinthY, 0]} receiveShadow raycast={() => null} />
      <mesh geometry={geometries.sideCap} material={materials.capMat} position={[-9.6, capY, 0]} receiveShadow raycast={() => null} />

      {/* ─── 2. RIGHT PERIMETER WALL (X = +9.6m) ─── */}
      <mesh geometry={geometries.sideWall} material={materials.wallBodyMat} position={[9.6, bodyY, 0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.sidePlinth} material={materials.wallTrimMat} position={[9.6, plinthY, 0]} receiveShadow raycast={() => null} />
      <mesh geometry={geometries.sideCap} material={materials.capMat} position={[9.6, capY, 0]} receiveShadow raycast={() => null} />

      {/* ─── 3. BACK PERIMETER WALL (Z = -13.0m) ─── */}
      <mesh geometry={geometries.frontWallHalf} material={materials.wallBodyMat} position={[-5.9, bodyY, -13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.frontPlinthHalf} material={materials.wallTrimMat} position={[-5.9, plinthY, -13.0]} receiveShadow raycast={() => null} />
      <mesh geometry={geometries.frontCapHalf} material={materials.capMat} position={[-5.9, capY, -13.0]} receiveShadow raycast={() => null} />

      <mesh geometry={geometries.frontWallHalf} material={materials.wallBodyMat} position={[5.9, bodyY, -13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.frontPlinthHalf} material={materials.wallTrimMat} position={[5.9, plinthY, -13.0]} receiveShadow raycast={() => null} />
      <mesh geometry={geometries.frontCapHalf} material={materials.capMat} position={[5.9, capY, -13.0]} receiveShadow raycast={() => null} />

      <mesh geometry={geometries.portalLintel} material={materials.capMat} position={[0, wallHeight - 0.20, -13.0]} receiveShadow raycast={() => null} />

      {/* ─── 4. FRONT PERIMETER WALL (Z = +13.0m) with Glass Entrance ─── */}
      <mesh geometry={geometries.frontWallHalf} material={materials.wallBodyMat} position={[-5.9, bodyY, 13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.frontPlinthHalf} material={materials.wallTrimMat} position={[-5.9, plinthY, 13.0]} receiveShadow raycast={() => null} />
      <mesh geometry={geometries.frontCapHalf} material={materials.capMat} position={[-5.9, capY, 13.0]} receiveShadow raycast={() => null} />

      <mesh geometry={geometries.frontWallHalf} material={materials.wallBodyMat} position={[5.9, bodyY, 13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.frontPlinthHalf} material={materials.wallTrimMat} position={[5.9, plinthY, 13.0]} receiveShadow raycast={() => null} />
      <mesh geometry={geometries.frontCapHalf} material={materials.capMat} position={[5.9, capY, 13.0]} receiveShadow raycast={() => null} />

      <mesh geometry={geometries.portalLintel} material={materials.capMat} position={[0, wallHeight - 0.20, 13.0]} receiveShadow raycast={() => null} />

      {/* Architectural Sliding Glass Doors */}
      <mesh geometry={geometries.glassPanel} material={materials.glassDoorMat} position={[-1.0, 1.25, 13.0]} raycast={() => null} />
      <mesh geometry={geometries.glassPanel} material={materials.glassDoorMat} position={[1.0, 1.25, 13.0]} raycast={() => null} />
      <mesh geometry={geometries.glassHandle} material={materials.glassFrameMat} position={[-0.15, 1.20, 13.03]} raycast={() => null} />
      <mesh geometry={geometries.glassHandle} material={materials.glassFrameMat} position={[0.15, 1.20, 13.03]} raycast={() => null} />

      {/* ─── 5. STRUCTURAL PILASTERS & ILLUMINATED LED COLUMNS ─── */}
      {/* 4 Corner Pilasters */}
      <mesh geometry={geometries.pilaster} material={materials.wallTrimMat} position={[-9.6, wallHeight / 2, -13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.pilaster} material={materials.wallTrimMat} position={[9.6, wallHeight / 2, -13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.pilaster} material={materials.wallTrimMat} position={[-9.6, wallHeight / 2, 13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.pilaster} material={materials.wallTrimMat} position={[9.6, wallHeight / 2, 13.0]} castShadow receiveShadow raycast={() => null} />

      {/* Side Midpoint Pilasters */}
      <mesh geometry={geometries.pilaster} material={materials.wallTrimMat} position={[-9.6, wallHeight / 2, 0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.pilaster} material={materials.wallTrimMat} position={[9.6, wallHeight / 2, 0]} castShadow receiveShadow raycast={() => null} />

      {/* Front Entrance Pilasters */}
      <mesh geometry={geometries.pilaster} material={materials.wallTrimMat} position={[-2.2, wallHeight / 2, 13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.pilaster} material={materials.wallTrimMat} position={[2.2, wallHeight / 2, 13.0]} castShadow receiveShadow raycast={() => null} />

      {/* Back Portal Pilasters */}
      <mesh geometry={geometries.pilaster} material={materials.wallTrimMat} position={[-2.2, wallHeight / 2, -13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.pilaster} material={materials.wallTrimMat} position={[2.2, wallHeight / 2, -13.0]} castShadow receiveShadow raycast={() => null} />

      {/* ─── EMBEDDED VERTICAL GLOWING WHITE LED STRIPS (matching reference image) ─── */}
      {/* Front Door Columns: Interior & Exterior strips */}
      <mesh geometry={geometries.verticalLedStrip} material={materials.ledStripMat} position={[-2.2, 1.50, 12.76]} raycast={() => null} />
      <mesh geometry={geometries.verticalLedStrip} material={materials.ledStripMat} position={[2.2, 1.50, 12.76]} raycast={() => null} />
      <mesh geometry={geometries.verticalLedStrip} material={materials.ledStripMat} position={[-2.2, 1.50, 13.24]} raycast={() => null} />
      <mesh geometry={geometries.verticalLedStrip} material={materials.ledStripMat} position={[2.2, 1.50, 13.24]} raycast={() => null} />

      {/* Back Door Columns */}
      <mesh geometry={geometries.verticalLedStrip} material={materials.ledStripMat} position={[-2.2, 1.50, -12.76]} raycast={() => null} />
      <mesh geometry={geometries.verticalLedStrip} material={materials.ledStripMat} position={[2.2, 1.50, -12.76]} raycast={() => null} />

      {/* Side Midpoint Columns */}
      <mesh geometry={geometries.verticalLedStrip} material={materials.ledStripMat} position={[-9.36, 1.50, 0]} rotation={[0, Math.PI / 2, 0]} raycast={() => null} />
      <mesh geometry={geometries.verticalLedStrip} material={materials.ledStripMat} position={[9.36, 1.50, 0]} rotation={[0, Math.PI / 2, 0]} raycast={() => null} />

      {/* Corner Columns Strips */}
      <mesh geometry={geometries.verticalLedStrip} material={materials.ledStripMat} position={[-9.36, 1.50, 12.76]} rotation={[0, Math.PI / 4, 0]} raycast={() => null} />
      <mesh geometry={geometries.verticalLedStrip} material={materials.ledStripMat} position={[9.36, 1.50, 12.76]} rotation={[0, -Math.PI / 4, 0]} raycast={() => null} />
      <mesh geometry={geometries.verticalLedStrip} material={materials.ledStripMat} position={[-9.36, 1.50, -12.76]} rotation={[0, 3 * Math.PI / 4, 0]} raycast={() => null} />
      <mesh geometry={geometries.verticalLedStrip} material={materials.ledStripMat} position={[9.36, 1.50, -12.76]} rotation={[0, -3 * Math.PI / 4, 0]} raycast={() => null} />

      {/* ─── 6. DOWNWARD ARCHITECTURAL WALL WASH SCONCES ─── */}
      {/* Left Wall Sconces */}
      <mesh geometry={geometries.sconceHousing} material={materials.sconceHousingMat} position={[-9.42, 2.75, -5.0]} rotation={[0, Math.PI / 2, 0]} raycast={() => null} />
      <mesh geometry={geometries.sconceLens} material={materials.sconceLensMat} position={[-9.42, 2.71, -5.0]} rotation={[0, Math.PI / 2, 0]} raycast={() => null} />
      <mesh geometry={geometries.sconceHousing} material={materials.sconceHousingMat} position={[-9.42, 2.75, 5.0]} rotation={[0, Math.PI / 2, 0]} raycast={() => null} />
      <mesh geometry={geometries.sconceLens} material={materials.sconceLensMat} position={[-9.42, 2.71, 5.0]} rotation={[0, Math.PI / 2, 0]} raycast={() => null} />

      {/* Right Wall Sconces */}
      <mesh geometry={geometries.sconceHousing} material={materials.sconceHousingMat} position={[9.42, 2.75, -5.0]} rotation={[0, -Math.PI / 2, 0]} raycast={() => null} />
      <mesh geometry={geometries.sconceLens} material={materials.sconceLensMat} position={[9.42, 2.71, -5.0]} rotation={[0, -Math.PI / 2, 0]} raycast={() => null} />
      <mesh geometry={geometries.sconceHousing} material={materials.sconceHousingMat} position={[9.42, 2.75, 5.0]} rotation={[0, -Math.PI / 2, 0]} raycast={() => null} />
      <mesh geometry={geometries.sconceLens} material={materials.sconceLensMat} position={[9.42, 2.71, 5.0]} rotation={[0, -Math.PI / 2, 0]} raycast={() => null} />

      {/* Back Wall Sconces */}
      <mesh geometry={geometries.sconceHousing} material={materials.sconceHousingMat} position={[-6.0, 2.75, -12.82]} raycast={() => null} />
      <mesh geometry={geometries.sconceLens} material={materials.sconceLensMat} position={[-6.0, 2.71, -12.82]} raycast={() => null} />
      <mesh geometry={geometries.sconceHousing} material={materials.sconceHousingMat} position={[6.0, 2.75, -12.82]} raycast={() => null} />
      <mesh geometry={geometries.sconceLens} material={materials.sconceLensMat} position={[6.0, 2.71, -12.82]} raycast={() => null} />

      {/* Front Wall Sconces */}
      <mesh geometry={geometries.sconceHousing} material={materials.sconceHousingMat} position={[-6.0, 2.75, 12.82]} rotation={[0, Math.PI, 0]} raycast={() => null} />
      <mesh geometry={geometries.sconceLens} material={materials.sconceLensMat} position={[-6.0, 2.71, 12.82]} rotation={[0, Math.PI, 0]} raycast={() => null} />
      <mesh geometry={geometries.sconceHousing} material={materials.sconceHousingMat} position={[6.0, 2.75, 12.82]} rotation={[0, Math.PI, 0]} raycast={() => null} />
      <mesh geometry={geometries.sconceLens} material={materials.sconceLensMat} position={[6.0, 2.71, 12.82]} rotation={[0, Math.PI, 0]} raycast={() => null} />
    </group>
  );
}
