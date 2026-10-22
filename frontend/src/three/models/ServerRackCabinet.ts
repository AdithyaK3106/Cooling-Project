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
  statusLedAmber: THREE.MeshBasicMaterial;
  haloTexture: THREE.CanvasTexture;
}

let sharedMaterials: SharedMaterials | null = null;

function getSharedMaterials(): SharedMaterials {
  if (!sharedMaterials) {
    // Generate soft radial halo texture for realistic underglow light bleed on floor
    const haloCanvas = document.createElement('canvas');
    haloCanvas.width = 128;
    haloCanvas.height = 128;
    const hCtx = haloCanvas.getContext('2d')!;
    const grad = hCtx.createRadialGradient(64, 64, 8, 64, 64, 64);
    grad.addColorStop(0, 'rgba(255, 255, 255, 1)');
    grad.addColorStop(0.35, 'rgba(255, 255, 255, 0.70)');
    grad.addColorStop(0.70, 'rgba(255, 255, 255, 0.22)');
    grad.addColorStop(1, 'rgba(255, 255, 255, 0)');
    hCtx.fillStyle = grad;
    hCtx.fillRect(0, 0, 128, 128);
    const haloTexture = new THREE.CanvasTexture(haloCanvas);

    sharedMaterials = {
      // Deep obsidian / graphite metal chassis
      chassisDark: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#101319'),
        metalness: 0.88,
        roughness: 0.28,
      }),
      // Beveled corner pillars and top trim
      chassisTrim: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#191d26'),
        metalness: 0.92,
        roughness: 0.22,
      }),
      // Brushed server blade steel
      serverFace: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#222834'),
        metalness: 0.80,
        roughness: 0.30,
      }),
      // Aluminum latch handles and mounting ears
      serverEar: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#475569'),
        metalness: 0.95,
        roughness: 0.18,
      }),
      // Recessed drive bay slots
      driveBay: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#131720'),
        metalness: 0.70,
        roughness: 0.40,
      }),
      // Top ventilation fan grilles
      fanGrille: new THREE.MeshStandardMaterial({
        color: new THREE.Color('#0d1016'),
        metalness: 0.90,
        roughness: 0.32,
      }),
      // Micro status LEDs
      statusLedGreen: new THREE.MeshBasicMaterial({
        color: new THREE.Color('#22c55e'),
      }),
      statusLedBlue: new THREE.MeshBasicMaterial({
        color: new THREE.Color('#38bdf8'),
      }),
      statusLedAmber: new THREE.MeshBasicMaterial({
        color: new THREE.Color('#f59e0b'),
      }),
      haloTexture,
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
      wallSide: new THREE.BoxGeometry(0.04, 2.42, 1.32),
      wallTopBottom: new THREE.BoxGeometry(1.16, 0.04, 1.36),
      wallBack: new THREE.BoxGeometry(1.08, 2.42, 0.03),
      pillarVertical: new THREE.BoxGeometry(0.065, 2.46, 0.065),
      crossbarHorizontal: new THREE.BoxGeometry(1.16, 0.08, 0.065),
      serverChassis: new THREE.BoxGeometry(1.02, 0.155, 1.24),
      serverEar: new THREE.BoxGeometry(0.035, 0.135, 0.035),
      driveBayGroup: new THREE.BoxGeometry(0.90, 0.125, 0.015),
      verticalLightPipe: new THREE.BoxGeometry(0.032, 2.26, 0.025),
      bladeIndicatorBar: new THREE.BoxGeometry(0.84, 0.012, 0.02),
      topBadgeBar: new THREE.BoxGeometry(0.92, 0.020, 0.03),
      fanRing: new THREE.RingGeometry(0.18, 0.30, 24),
      sidePanelRecess: new THREE.BoxGeometry(0.01, 2.16, 1.16),
      floorHalo: new THREE.PlaneGeometry(2.30, 2.50),
      microLed: new THREE.BoxGeometry(0.018, 0.018, 0.015),
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

  // Dynamic risk accent material (vibrant luminescence)
  const riskAccentMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#06b6d4'),
    emissive: new THREE.Color('#06b6d4'),
    emissiveIntensity: 1.2,
    roughness: 0.15,
    metalness: 0.40,
  });
  (riskAccentMat as any).__isRiskAccentMaterial = true;

  // Luminescent floor underglow reflection
  const floorGlowMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color('#06b6d4'),
    emissive: new THREE.Color('#06b6d4'),
    emissiveIntensity: 0.85,
    alphaMap: mats.haloTexture,
    transparent: true,
    opacity: 0.65,
    roughness: 0.30,
    metalness: 0.10,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  (floorGlowMat as any).__isRiskAccentMaterial = true;

  // 1. CHASSIS SHELL (Dark Graphite Metal)
  // Left side wall
  const leftWall = new THREE.Mesh(geoms.wallSide, mats.chassisDark);
  leftWall.position.set(-0.56, 1.23, 0);
  leftWall.castShadow = true;
  leftWall.receiveShadow = true;
  cabinet.add(leftWall);

  // Right side wall
  const rightWall = new THREE.Mesh(geoms.wallSide, mats.chassisDark);
  rightWall.position.set(0.56, 1.23, 0);
  rightWall.castShadow = true;
  rightWall.receiveShadow = true;
  cabinet.add(rightWall);

  // Top roof
  const roof = new THREE.Mesh(geoms.wallTopBottom, mats.chassisDark);
  roof.position.set(0, 2.44, 0);
  cabinet.add(roof);

  // Bottom plinth footer
  const footer = new THREE.Mesh(geoms.wallTopBottom, mats.chassisDark);
  footer.position.set(0, 0.02, 0);
  cabinet.add(footer);

  // Luminescent floor underglow reflection pool around the rack base
  const floorGlow = new THREE.Mesh(geoms.floorHalo, floorGlowMat);
  floorGlow.rotation.x = -Math.PI / 2;
  floorGlow.position.set(0, 0.008, 0);
  floorGlow.userData = { isRiskAccent: true };
  cabinet.add(floorGlow);

  // Back panel
  const back = new THREE.Mesh(geoms.wallBack, mats.chassisDark);
  back.position.set(0, 1.23, -0.665);
  cabinet.add(back);

  // Recessed exterior side panel details (left & right)
  const leftRecess = new THREE.Mesh(geoms.sidePanelRecess, mats.chassisTrim);
  leftRecess.position.set(-0.582, 1.23, 0);
  cabinet.add(leftRecess);

  const rightRecess = new THREE.Mesh(geoms.sidePanelRecess, mats.chassisTrim);
  rightRecess.position.set(0.582, 1.23, 0);
  cabinet.add(rightRecess);

  // 2. BEVELED CORNER PILLARS & FRONT FRAME
  // Front-left pillar
  const flPillar = new THREE.Mesh(geoms.pillarVertical, mats.chassisTrim);
  flPillar.position.set(-0.55, 1.23, 0.65);
  cabinet.add(flPillar);

  // Front-right pillar
  const frPillar = new THREE.Mesh(geoms.pillarVertical, mats.chassisTrim);
  frPillar.position.set(0.55, 1.23, 0.65);
  cabinet.add(frPillar);

  // Top front crossbar
  const topBar = new THREE.Mesh(geoms.crossbarHorizontal, mats.chassisTrim);
  topBar.position.set(0, 2.42, 0.65);
  cabinet.add(topBar);

  // Bottom front crossbar
  const bottomBar = new THREE.Mesh(geoms.crossbarHorizontal, mats.chassisTrim);
  bottomBar.position.set(0, 0.04, 0.65);
  cabinet.add(bottomBar);

  // Top illuminated header badge
  const topBadge = new THREE.Mesh(geoms.topBadgeBar, riskAccentMat);
  topBadge.position.set(0, 2.38, 0.675);
  topBadge.userData = { isRiskAccent: true };
  cabinet.add(topBadge);

  // 3. DUAL VERTICAL ACCENT LIGHT PIPES (Left & Right Front Corners)
  const leftLightPipe = new THREE.Mesh(geoms.verticalLightPipe, riskAccentMat);
  leftLightPipe.position.set(-0.515, 1.22, 0.665);
  leftLightPipe.userData = { isRiskAccent: true };
  cabinet.add(leftLightPipe);

  const rightLightPipe = new THREE.Mesh(geoms.verticalLightPipe, riskAccentMat);
  rightLightPipe.position.set(0.515, 1.22, 0.665);
  rightLightPipe.userData = { isRiskAccent: true };
  cabinet.add(rightLightPipe);

  // 4. SERVER HARDWARE BLADES (12 Stacked Server Units)
  const NUM_BLADES = 12;
  const startY = 0.18;
  const bladeSpacing = 0.182;

  for (let b = 0; b < NUM_BLADES; b++) {
    const bladeY = startY + b * bladeSpacing;

    // Server blade body
    const bladeChassis = new THREE.Mesh(geoms.serverChassis, mats.serverFace);
    bladeChassis.position.set(0, bladeY, 0.02);
    cabinet.add(bladeChassis);

    // Left handle / mounting ear
    const leftEar = new THREE.Mesh(geoms.serverEar, mats.serverEar);
    leftEar.position.set(-0.50, bladeY, 0.655);
    cabinet.add(leftEar);

    // Right handle / mounting ear
    const rightEar = new THREE.Mesh(geoms.serverEar, mats.serverEar);
    rightEar.position.set(0.50, bladeY, 0.655);
    cabinet.add(rightEar);

    // Front drive bays / ventilation grille plate
    const driveBays = new THREE.Mesh(geoms.driveBayGroup, mats.driveBay);
    driveBays.position.set(0, bladeY, 0.65);
    cabinet.add(driveBays);

    // Micro activity LEDs on server face (Power green, Activity blue, Amber load)
    const ledPower = new THREE.Mesh(geoms.microLed, mats.statusLedGreen);
    ledPower.position.set(-0.46, bladeY, 0.660);
    cabinet.add(ledPower);

    const ledActivity = new THREE.Mesh(geoms.microLed, b % 3 === 0 ? mats.statusLedAmber : mats.statusLedBlue);
    ledActivity.position.set(-0.435, bladeY, 0.660);
    cabinet.add(ledActivity);

    // Subtle horizontal risk accent indicator along each server blade
    const bladeAccent = new THREE.Mesh(geoms.bladeIndicatorBar, riskAccentMat);
    bladeAccent.position.set(0, bladeY - 0.065, 0.658);
    bladeAccent.userData = { isRiskAccent: true };
    cabinet.add(bladeAccent);
  }

  // 5. TOP ROOF VENTILATION FANS
  const fan1 = new THREE.Mesh(geoms.fanRing, mats.fanGrille);
  fan1.rotation.x = -Math.PI / 2;
  fan1.position.set(-0.25, 2.462, 0);
  cabinet.add(fan1);

  const fan2 = new THREE.Mesh(geoms.fanRing, mats.fanGrille);
  fan2.rotation.x = -Math.PI / 2;
  fan2.position.set(0.25, 2.462, 0);
  cabinet.add(fan2);

  // Tag cabinet container
  cabinet.userData = {
    rackId,
    isDetailedCabinet: true,
    accentMaterial: riskAccentMat,
    floorGlowMaterial: floorGlowMat,
  };

  return cabinet;
}
