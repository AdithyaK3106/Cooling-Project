import { useMemo } from 'react';
import * as THREE from 'three';

/**
 * PerimeterWall: Minimal, clean, modern industrial data-center perimeter wall
 * enclosing the server-rack floor area.
 * 
 * Inspired by enterprise data-center reference architecture:
 * - Straight, clean industrial geometry defining the rack zone boundary
 * - Dark/neutral visual language matching the existing floor and chassis
 * - Continuous plinth footer, recessed wall panels, and beveled top coping cap
 * - Structural corner/intermediate pilasters for authentic architectural rhythm
 * - Non-blocking portal openings on front/back for unobstructed camera visibility
 * - Disabled raycasting to guarantee 100% hover & click transparency for racks
 */
export function PerimeterWall() {
  const { materials, geometries } = useMemo(() => {
    // Industrial dark/neutral material palette
    const wallBodyMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#1e222b'),
      metalness: 0.22,
      roughness: 0.58,
    });

    const trimMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#14171e'),
      metalness: 0.35,
      roughness: 0.35,
    });

    const capMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color('#282d38'),
      metalness: 0.45,
      roughness: 0.28,
    });

    // Wall dimensions: Enclosing 5x5 rack cluster (±9.6m on X, ±13.0m on Z)
    const wallHeight = 2.60;
    const wallThick = 0.26;
    const plinthHeight = 0.16;
    const capHeight = 0.08;

    const sideLength = 26.0; // Z span (-13 to +13)
    const frontHalfLength = 7.4; // X span from corner ±9.6 to portal ±2.2

    return {
      materials: { wallBodyMat, trimMat, capMat },
      geometries: {
        // Left & right continuous walls
        sideWall: new THREE.BoxGeometry(wallThick, wallHeight - plinthHeight - capHeight, sideLength),
        sidePlinth: new THREE.BoxGeometry(wallThick + 0.04, plinthHeight, sideLength),
        sideCap: new THREE.BoxGeometry(wallThick + 0.06, capHeight, sideLength + 0.06),

        // Front & back segmented walls with central portal openings
        frontWallHalf: new THREE.BoxGeometry(frontHalfLength, wallHeight - plinthHeight - capHeight, wallThick),
        frontPlinthHalf: new THREE.BoxGeometry(frontHalfLength, plinthHeight, wallThick + 0.04),
        frontCapHalf: new THREE.BoxGeometry(frontHalfLength + 0.03, capHeight, wallThick + 0.06),

        // Portal lintel beam spanning over entrance
        portalLintel: new THREE.BoxGeometry(4.4, 0.35, wallThick + 0.02),

        // Structural corner & intermediate pilasters
        pilaster: new THREE.BoxGeometry(0.38, wallHeight + 0.06, 0.38),
      },
    };
  }, []);

  const wallHeight = 2.60;
  const bodyY = 0.16 + (wallHeight - 0.16 - 0.08) / 2; // ~1.34m
  const capY = wallHeight - 0.04;                      // ~2.56m
  const plinthY = 0.08;                               // ~0.08m

  return (
    <group name="DataCenterPerimeterWall">
      {/* ─── 1. LEFT PERIMETER WALL (X = -9.6m) ─── */}
      <mesh geometry={geometries.sideWall} material={materials.wallBodyMat} position={[-9.6, bodyY, 0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.sidePlinth} material={materials.trimMat} position={[-9.6, plinthY, 0]} receiveShadow raycast={() => null} />
      <mesh geometry={geometries.sideCap} material={materials.capMat} position={[-9.6, capY, 0]} receiveShadow raycast={() => null} />

      {/* ─── 2. RIGHT PERIMETER WALL (X = +9.6m) ─── */}
      <mesh geometry={geometries.sideWall} material={materials.wallBodyMat} position={[9.6, bodyY, 0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.sidePlinth} material={materials.trimMat} position={[9.6, plinthY, 0]} receiveShadow raycast={() => null} />
      <mesh geometry={geometries.sideCap} material={materials.capMat} position={[9.6, capY, 0]} receiveShadow raycast={() => null} />

      {/* ─── 3. BACK PERIMETER WALL (Z = -13.0m) with Central Portal ─── */}
      {/* Back-left wing */}
      <mesh geometry={geometries.frontWallHalf} material={materials.wallBodyMat} position={[-5.9, bodyY, -13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.frontPlinthHalf} material={materials.trimMat} position={[-5.9, plinthY, -13.0]} receiveShadow raycast={() => null} />
      <mesh geometry={geometries.frontCapHalf} material={materials.capMat} position={[-5.9, capY, -13.0]} receiveShadow raycast={() => null} />
      {/* Back-right wing */}
      <mesh geometry={geometries.frontWallHalf} material={materials.wallBodyMat} position={[5.9, bodyY, -13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.frontPlinthHalf} material={materials.trimMat} position={[5.9, plinthY, -13.0]} receiveShadow raycast={() => null} />
      <mesh geometry={geometries.frontCapHalf} material={materials.capMat} position={[5.9, capY, -13.0]} receiveShadow raycast={() => null} />
      {/* Back portal lintel */}
      <mesh geometry={geometries.portalLintel} material={materials.capMat} position={[0, wallHeight - 0.175, -13.0]} receiveShadow raycast={() => null} />

      {/* ─── 4. FRONT PERIMETER WALL (Z = +13.0m) with Central Portal ─── */}
      {/* Front-left wing */}
      <mesh geometry={geometries.frontWallHalf} material={materials.wallBodyMat} position={[-5.9, bodyY, 13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.frontPlinthHalf} material={materials.trimMat} position={[-5.9, plinthY, 13.0]} receiveShadow raycast={() => null} />
      <mesh geometry={geometries.frontCapHalf} material={materials.capMat} position={[-5.9, capY, 13.0]} receiveShadow raycast={() => null} />
      {/* Front-right wing */}
      <mesh geometry={geometries.frontWallHalf} material={materials.wallBodyMat} position={[5.9, bodyY, 13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.frontPlinthHalf} material={materials.trimMat} position={[5.9, plinthY, 13.0]} receiveShadow raycast={() => null} />
      <mesh geometry={geometries.frontCapHalf} material={materials.capMat} position={[5.9, capY, 13.0]} receiveShadow raycast={() => null} />
      {/* Front portal lintel */}
      <mesh geometry={geometries.portalLintel} material={materials.capMat} position={[0, wallHeight - 0.175, 13.0]} receiveShadow raycast={() => null} />

      {/* ─── 5. STRUCTURAL PILASTERS / COLUMNS ─── */}
      {/* 4 Corners */}
      <mesh geometry={geometries.pilaster} material={materials.trimMat} position={[-9.6, wallHeight / 2, -13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.pilaster} material={materials.trimMat} position={[9.6, wallHeight / 2, -13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.pilaster} material={materials.trimMat} position={[-9.6, wallHeight / 2, 13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.pilaster} material={materials.trimMat} position={[9.6, wallHeight / 2, 13.0]} castShadow receiveShadow raycast={() => null} />
      {/* Side Midpoints */}
      <mesh geometry={geometries.pilaster} material={materials.trimMat} position={[-9.6, wallHeight / 2, 0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.pilaster} material={materials.trimMat} position={[9.6, wallHeight / 2, 0]} castShadow receiveShadow raycast={() => null} />
      {/* Portal Flanking Columns */}
      <mesh geometry={geometries.pilaster} material={materials.trimMat} position={[-2.2, wallHeight / 2, -13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.pilaster} material={materials.trimMat} position={[2.2, wallHeight / 2, -13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.pilaster} material={materials.trimMat} position={[-2.2, wallHeight / 2, 13.0]} castShadow receiveShadow raycast={() => null} />
      <mesh geometry={geometries.pilaster} material={materials.trimMat} position={[2.2, wallHeight / 2, 13.0]} castShadow receiveShadow raycast={() => null} />
    </group>
  );
}
