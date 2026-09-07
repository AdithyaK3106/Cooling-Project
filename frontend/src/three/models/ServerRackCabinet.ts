import * as THREE from 'three';

// Shared materials across all 25 racks for maximum performance (60 FPS)
interface SharedMaterials {
  chassisDark: THREE.MeshStandardMaterial;
  chassisTrim: THREE.MeshStandardMaterial;
  serverFace: THREE.MeshStandardMaterial;
  serverEar: THREE.MeshStandardMaterial;
  driveBay: THREE.MeshStandardMaterial;
  fanGrille: THREE.MeshStandardMaterial;
  statusLedGreen: THREE.MeshBasicMaterial;
  statusLedBlue: THREE.MeshBasicMaterial;
}

let sharedMaterials: SharedMaterials | null = null;

function getSharedMaterials(): SharedMaterials {
  if (!sharedMaterials) {
    sharedMaterials = {
      // Deep graphite / obsidian metal for the main chassis
      chassisDark: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#121418'),
        metalness: 0.86,
        roughness: 0.26,
      }),
      // Slightly lighter charcoal metal for beveled front pillars and frames
      chassisTrim: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#1c1f26'),
        metalness: 0.90,
        roughness: 0.22,
      }),
      // Brushed server blade steel
      serverFace: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#22262e'),
        metalness: 0.78,
        roughness: 0.32,
      }),
      // Aluminum latch handles and mounting ears
      serverEar: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#3c434f'),
        metalness: 0.92,
        roughness: 0.20,
      }),
      // Recessed drive bay slots
      driveBay: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#14171d'),
        metalness: 0.65,
        roughness: 0.45,
      }),
      // Top ventilation fan grilles
      fanGrille: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#101216'),
        metalness: 0.88,
        roughness: 0.35,
      }),
      // Tiny micro activity LEDs (green power indicator)
      statusLedGreen: new THREE.MeshBasicMaterial({
        color: new THREE.Color('#22c55e'),
      }),
      // Tiny micro activity LEDs (blue drive indicator)
      statusLedBlue: new THREE.MeshBasicMaterial({
        color: new THREE.Color('#38bdf8'),
      }),
    };
  }
  return sharedMaterials;
}

// Shared geometries across all 25 racks
interface SharedGeometries {
  wallSide: THREE.BoxGeometry;
  wallTopBottom: THREE.BoxGeometry;
  wallBack: THREE.BoxGeometry;
  pillarVertical: THREE.BoxGeometry;
  crossbarHorizontal: THREE.BoxGeometry;
  serverChassis: THREE.BoxGeometry;
  serverEar: THREE.BoxGeometry;
  driveBayGroup: THREE.BoxGeometry;
  verticalLightPipe: THREE.BoxGeometry;
  bladeIndicatorBar: THREE.BoxGeometry;
  topBadgeBar: THREE.BoxGeometry;
  fanRing: THREE.RingGeometry;
  sidePanelRecess: THREE.BoxGeometry;
  floorHalo: THREE.PlaneGeometry;
  microLed: THREE.BoxGeometry;
}

let sharedGeometries: SharedGeometries | null = null;

function getSharedGeometries(): SharedGeometries {
  if (!sharedGeometries) {
    sharedGeometries = {
      wallSide: new THREE.BoxGeometry(0.04, 2.36, 1.26),
      wallTopBottom: new THREE.BoxGeometry(1.10, 0.04, 1.30),
      wallBack: new THREE.BoxGeometry(1.02, 2.36, 0.03),
      pillarVertical: new THREE.BoxGeometry(0.06, 2.40, 0.06),
      crossbarHorizontal: new THREE.BoxGeometry(1.10, 0.08, 0.06),
      serverChassis: new THREE.BoxGeometry(0.96, 0.15, 1.18),
      serverEar: new THREE.BoxGeometry(0.03, 0.13, 0.03),
      driveBayGroup: new THREE.BoxGeometry(0.84, 0.12, 0.015),
      verticalLightPipe: new THREE.BoxGeometry(0.02, 2.18, 0.018),
      bladeIndicatorBar: new THREE.BoxGeometry(0.78, 0.008, 0.015),
      topBadgeBar: new THREE.BoxGeometry(0.86, 0.014, 0.02),
      fanRing: new THREE.RingGeometry(0.18, 0.28, 24),
      sidePanelRecess: new THREE.BoxGeometry(0.01, 2.10, 1.10),
      floorHalo: new THREE.PlaneGeometry(1.26, 1.46),
      microLed: new THREE.BoxGeometry(0.014, 0.014, 0.012),
    };
  }
  return sharedGeometries;
}

/**
 * Creates a highly detailed, realistic enterprise data-center server cabinet.
 * Matches standard 42U proportions with graphite chassis, horizontal server blades,
 * drive caddies, handle ears, top exhaust fans, and dedicated risk accent lighting strips.
 */
export function createDetailedServerRack(rackId: string): THREE.Group {
  const mats = getSharedMaterials();
  const geoms = getSharedGeometries();

  const cabinet = new THREE.Group();
  cabinet.name = `EnterpriseCabinet_${rackId}`;

  // Unique risk accent material per rack for independent dynamic color state
  const riskAccentMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#06b6d4'),
    emissive: new THREE.Color('#06b6d4'),
    emissiveIntensity: 0.70,
    roughness: 0.20,
    metalness: 0.40,
  });
  (riskAccentMat as any).__isRiskAccentMaterial = true;

  // Soft semi-transparent floor underglow reflection
  const floorGlowMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#06b6d4'),
    emissive: new THREE.Color('#06b6d4'),
    emissiveIntensity: 0.35,
    transparent: true,
    opacity: 0.24,
    roughness: 0.40,
    metalness: 0.10,
    depthWrite: false,
  });
  (floorGlowMat as any).__isRiskAccentMaterial = true;

  // 1. CHASSIS SHELL (Dark Graphite Metal)
  // Left side wall
  const leftWall = new THREE.Mesh(geoms.wallSide, mats.chassisDark);
  leftWall.position.set(-0.53, 1.20, 0);
  leftWall.castShadow = true;
  leftWall.receiveShadow = true;
  cabinet.add(leftWall);

  // Right side wall
  const rightWall = new THREE.Mesh(geoms.wallSide, mats.chassisDark);
  rightWall.position.set(0.53, 1.20, 0);
  rightWall.castShadow = true;
  rightWall.receiveShadow = true;
  cabinet.add(rightWall);

  // Top roof
  const roof = new THREE.Mesh(geoms.wallTopBottom, mats.chassisDark);
  roof.position.set(0, 2.38, 0);
  cabinet.add(roof);

  // Bottom plinth footer
  const footer = new THREE.Mesh(geoms.wallTopBottom, mats.chassisDark);
  footer.position.set(0, 0.02, 0);
  cabinet.add(footer);

  // Subtle floor underglow / light reflection around the rack base
  const floorGlow = new THREE.Mesh(geoms.floorHalo, floorGlowMat);
  floorGlow.rotation.x = -Math.PI / 2;
  floorGlow.position.set(0, 0.006, 0);
  floorGlow.userData = { isRiskAccent: true };
  cabinet.add(floorGlow);

  // Back panel
  const back = new THREE.Mesh(geoms.wallBack, mats.chassisDark);
  back.position.set(0, 1.20, -0.635);
  cabinet.add(back);

  // Recessed exterior side panel details (left & right)
  const leftRecess = new THREE.Mesh(geoms.sidePanelRecess, mats.chassisTrim);
  leftRecess.position.set(-0.552, 1.20, 0);
  cabinet.add(leftRecess);

  const rightRecess = new THREE.Mesh(geoms.sidePanelRecess, mats.chassisTrim);
  rightRecess.position.set(0.552, 1.20, 0);
  cabinet.add(rightRecess);

  // 2. BEVELED CORNER PILLARS & FRONT FRAME
  // Front-left pillar
  const flPillar = new THREE.Mesh(geoms.pillarVertical, mats.chassisTrim);
  flPillar.position.set(-0.52, 1.20, 0.62);
  cabinet.add(flPillar);

  // Front-right pillar
  const frPillar = new THREE.Mesh(geoms.pillarVertical, mats.chassisTrim);
  frPillar.position.set(0.52, 1.20, 0.62);
  cabinet.add(frPillar);

  // Top front crossbar
  const topBar = new THREE.Mesh(geoms.crossbarHorizontal, mats.chassisTrim);
  topBar.position.set(0, 2.36, 0.62);
  cabinet.add(topBar);

  // Bottom front crossbar
  const bottomBar = new THREE.Mesh(geoms.crossbarHorizontal, mats.chassisTrim);
  bottomBar.position.set(0, 0.04, 0.62);
  cabinet.add(bottomBar);

  // Top illuminated header badge / beacon
  const topBadge = new THREE.Mesh(geoms.topBadgeBar, riskAccentMat);
  topBadge.position.set(0, 2.32, 0.64);
  topBadge.userData = { isRiskAccent: true };
  cabinet.add(topBadge);

  // 3. DUAL VERTICAL ACCENT LIGHT PIPES (Left & Right)
  const leftLightPipe = new THREE.Mesh(geoms.verticalLightPipe, riskAccentMat);
  leftLightPipe.position.set(-0.485, 1.19, 0.63);
  leftLightPipe.userData = { isRiskAccent: true };
  cabinet.add(leftLightPipe);

  const rightLightPipe = new THREE.Mesh(geoms.verticalLightPipe, riskAccentMat);
  rightLightPipe.position.set(0.485, 1.19, 0.63);
  rightLightPipe.userData = { isRiskAccent: true };
  cabinet.add(rightLightPipe);

  // 4. SERVER HARDWARE BLADES (12 Stacked Server Units)
  const NUM_BLADES = 12;
  const startY = 0.16;
  const bladeSpacing = 0.178;

  for (let b = 0; b < NUM_BLADES; b++) {
    const bladeY = startY + b * bladeSpacing;

    // Server blade body
    const bladeChassis = new THREE.Mesh(geoms.serverChassis, mats.serverFace);
    bladeChassis.position.set(0, bladeY, 0.02);
    cabinet.add(bladeChassis);

    // Left handle / mounting ear
    const leftEar = new THREE.Mesh(geoms.serverEar, mats.serverEar);
    leftEar.position.set(-0.47, bladeY, 0.625);
    cabinet.add(leftEar);

    // Right handle / mounting ear
    const rightEar = new THREE.Mesh(geoms.serverEar, mats.serverEar);
    rightEar.position.set(0.47, bladeY, 0.625);
    cabinet.add(rightEar);

    // Front drive bays / ventilation grille plate
    const driveBays = new THREE.Mesh(geoms.driveBayGroup, mats.driveBay);
    driveBays.position.set(0, bladeY, 0.62);
    cabinet.add(driveBays);

    // Tiny micro activity LEDs on server face (Power green, Activity blue)
    const ledPower = new THREE.Mesh(geoms.microLed, mats.statusLedGreen);
    ledPower.position.set(-0.432, bladeY, 0.629);
    cabinet.add(ledPower);

    const ledActivity = new THREE.Mesh(geoms.microLed, mats.statusLedBlue);
    ledActivity.position.set(-0.412, bladeY, 0.629);
    cabinet.add(ledActivity);

    // Subtle horizontal risk accent indicator along each server blade
    const bladeAccent = new THREE.Mesh(geoms.bladeIndicatorBar, riskAccentMat);
    bladeAccent.position.set(0, bladeY - 0.065, 0.628);
    bladeAccent.userData = { isRiskAccent: true };
    cabinet.add(bladeAccent);
  }

  // 5. TOP ROOF VENTILATION FANS
  const fan1 = new THREE.Mesh(geoms.fanRing, mats.fanGrille);
  fan1.rotation.x = -Math.PI / 2;
  fan1.position.set(-0.25, 2.402, 0);
  cabinet.add(fan1);

  const fan2 = new THREE.Mesh(geoms.fanRing, mats.fanGrille);
  fan2.rotation.x = -Math.PI / 2;
  fan2.position.set(0.25, 2.402, 0);
  cabinet.add(fan2);

  // Tag cabinet container
  cabinet.userData = {
    rackId,
    isDetailedCabinet: true,
    accentMaterial: riskAccentMat,
  };

  return cabinet;
}
