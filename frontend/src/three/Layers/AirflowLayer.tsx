import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// Professional enterprise airflow visualization:
// Stratified, clean stream count (20 cleanly spaced streams per rack row = 100 streams)
const STREAMS_PER_ROW = 20;
const STREAM_COUNT = STREAMS_PER_ROW * 5;

// High sample resolution for silky-smooth, continuous curve rendering
const POINTS_PER_STREAM = 32;
const SEGMENTS_PER_STREAM = POINTS_PER_STREAM - 1;
const TOTAL_VERTICES = STREAM_COUNT * SEGMENTS_PER_STREAM * 2;

// Datacenter 5x5 rack grid coordinates:
// Racks are positioned at X: -7.0, -3.5, 0.0, 3.5, 7.0
// The 5 rack rows are centered at Z: -10.0, -5.0, 0.0, 5.0, 10.0
// Rack row boundaries along X: from -7.6 (inlet before Rack 1) to +7.6 (exhaust after Rack 5)
const RACK_ROWS_Z = [-10.0, -5.0, 0.0, 5.0, 10.0];

const RACK_INLET_X = -7.6;
const RACK_EXHAUST_X = 7.6;
const RACK_SPAN_X = RACK_EXHAUST_X - RACK_INLET_X;

// Light-blue / cyan-blue tone (#0ea5e9: rgb(14, 165, 233))
const COLOR_R = 14 / 255;
const COLOR_G = 165 / 255;
const COLOR_B = 233 / 255;

interface StreamData {
  xHead: number;       // Front tip position along X
  length: number;      // Streamline length along X (bridges between adjacent racks)
  speed: number;       // Forward flow speed along +X
  baseY: number;       // Stratified tier height inside rack volume (0.45 - 1.55)
  baseZ: number;       // Stratified lane depth inside rack row
  spatialK1: number;   // Primary broad aerodynamic wave frequency
  spatialK2: number;   // Secondary harmonic frequency (subtle fluid softness)
  spatialKz: number;   // Lateral serpentine wave frequency
  rippleFreq: number;  // Fluid undulation ripple frequency
  ampY1: number;       // Primary vertical wave amplitude
  ampY2: number;       // Secondary vertical wave amplitude
  ampZ1: number;       // Primary lateral wave amplitude
  ampZ2: number;       // Secondary lateral wave amplitude
  phaseY1: number;     // Vertical phase offset
  phaseY2: number;     // Secondary phase offset
  phaseZ: number;      // Lateral phase offset
}

export function AirflowLayer() {
  const lineRef = useRef<THREE.LineSegments>(null);

  // Pre-generate cleanly stratified, non-cluttered airflow streamlines through the racks
  const streams = useMemo<StreamData[]>(() => {
    const arr: StreamData[] = [];

    // Stratified height tiers inside the 42U rack chassis: lower (0.50), mid (0.90), upper (1.30), top (1.50)
    const heightTiers = [0.50, 0.75, 1.00, 1.25, 1.50];
    // Stratified depth lanes across rack depth (~0.8 units): front (-0.22), center (0.0), rear (+0.22)
    const depthLanes = [-0.22, 0.0, 0.22];

    for (let rowIdx = 0; rowIdx < RACK_ROWS_Z.length; rowIdx++) {
      const rowZ = RACK_ROWS_Z[rowIdx];

      for (let s = 0; s < STREAMS_PER_ROW; s++) {
        // Cleanly distribute across height tiers and depth lanes with subtle organic jitter
        const tierY = heightTiers[s % heightTiers.length];
        const laneZ = depthLanes[Math.floor(s / heightTiers.length) % depthLanes.length];

        const baseY = tierY + (Math.random() - 0.5) * 0.10;
        const baseZ = rowZ + laneZ + (Math.random() - 0.5) * 0.08;

        // Long, elegant streamlines (2.6 - 3.6 units) bridging rack to rack
        const length = 2.6 + Math.random() * 1.0;

        // Stagger initial xHead evenly along the rack row
        const xHead = RACK_INLET_X + (s / STREAMS_PER_ROW) * RACK_SPAN_X + (Math.random() - 0.5) * 1.5;

        // Smooth, restrained flow speed
        const speed = 0.025 + Math.random() * 0.014;

        // Gentle, broad aerodynamic wavelength (spanning 3.0 to 4.2 units)
        const wavelength = 3.0 + Math.random() * 1.2;
        const spatialK1 = (Math.PI * 2) / wavelength;
        const spatialK2 = spatialK1 * 1.62;
        const spatialKz = spatialK1 * 0.88;

        arr.push({
          xHead,
          length,
          speed,
          baseY,
          baseZ,
          spatialK1,
          spatialK2,
          spatialKz,
          rippleFreq: 1.1 + Math.random() * 0.6,
          // Subtle, restrained amplitudes for clean, professional curves
          ampY1: 0.048 + Math.random() * 0.016,
          ampY2: 0.012 + Math.random() * 0.006,
          ampZ1: 0.038 + Math.random() * 0.014,
          ampZ2: 0.010 + Math.random() * 0.005,
          phaseY1: Math.random() * Math.PI * 2,
          phaseY2: Math.random() * Math.PI * 2,
          phaseZ: Math.random() * Math.PI * 2,
        });
      }
    }

    return arr;
  }, []);

  // Pre-allocate typed arrays for LineSegments geometry (positions + RGBA vertex colors)
  const { positions, colors, geometry } = useMemo(() => {
    const posArr = new Float32Array(TOTAL_VERTICES * 3);
    const colArr = new Float32Array(TOTAL_VERTICES * 4);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colArr, 4));

    return { positions: posArr, colors: colArr, geometry: geo };
  }, []);

  useFrame(({ clock }) => {
    if (!lineRef.current) return;

    const t = clock.getElapsedTime();
    let posPtr = 0;
    let colPtr = 0;

    for (let sIdx = 0; sIdx < STREAM_COUNT; sIdx++) {
      const s = streams[sIdx];

      // Advance streamline horizontally along +X direction through the racks
      s.xHead += s.speed;

      // Wrap around seamlessly when streamline tail fully exits the rack row
      if (s.xHead - s.length > RACK_EXHAUST_X) {
        s.xHead = RACK_INLET_X;
      }

      const xTail = s.xHead - s.length;

      // Calculate visible portion of the stream within rack row bounds
      const xVisStart = Math.max(RACK_INLET_X, xTail);
      const xVisEnd = Math.min(RACK_EXHAUST_X, s.xHead);
      const visSpan = xVisEnd - xVisStart;

      if (visSpan <= 0.02) {
        // Streamline is outside row bounds: collapse to degenerate invisible vertices
        for (let seg = 0; seg < SEGMENTS_PER_STREAM; seg++) {
          positions[posPtr] = RACK_INLET_X;
          positions[posPtr + 1] = s.baseY;
          positions[posPtr + 2] = s.baseZ;
          posPtr += 3;
          positions[posPtr] = RACK_INLET_X;
          positions[posPtr + 1] = s.baseY;
          positions[posPtr + 2] = s.baseZ;
          posPtr += 3;

          colors[colPtr] = COLOR_R;
          colors[colPtr + 1] = COLOR_G;
          colors[colPtr + 2] = COLOR_B;
          colors[colPtr + 3] = 0;
          colPtr += 4;
          colors[colPtr] = COLOR_R;
          colors[colPtr + 1] = COLOR_G;
          colors[colPtr + 2] = COLOR_B;
          colors[colPtr + 3] = 0;
          colPtr += 4;
        }
        continue;
      }

      // Calculate fluid sinusoidal wave points and per-vertex alpha gradient
      for (let seg = 0; seg < SEGMENTS_PER_STREAM; seg++) {
        // Point 1
        const u1 = seg / SEGMENTS_PER_STREAM;
        const x1 = xVisStart + u1 * visSpan;
        const localU1 = Math.max(0, Math.min(1, (x1 - xTail) / s.length));
        // Aerodynamic envelope: soft tapering at tail, steady body, crisp leader
        const env1 = 0.45 + 0.55 * Math.sin(localU1 * Math.PI);

        const distFromHead1 = s.xHead - x1;
        const thetaY1_1 = s.spatialK1 * distFromHead1 + s.rippleFreq * t + s.phaseY1;
        const thetaY2_1 = s.spatialK2 * distFromHead1 + s.rippleFreq * 1.35 * t + s.phaseY2;
        const thetaZ_1 = s.spatialKz * distFromHead1 + s.rippleFreq * 0.9 * t + s.phaseZ;

        const y1 = s.baseY + env1 * (s.ampY1 * Math.sin(thetaY1_1) + s.ampY2 * Math.sin(thetaY2_1));
        const z1 = s.baseZ + env1 * (s.ampZ1 * Math.cos(thetaZ_1) + s.ampZ2 * Math.sin(thetaY1_1));

        // Soft gradient alpha along stream: feathery tail fading to bright leader head
        const streamAlpha1 = Math.sin(Math.pow(localU1, 0.75) * Math.PI);
        // Smooth entrance and exit fade at rack row boundaries
        const boundFade1 = Math.min(1, Math.min(x1 - RACK_INLET_X, RACK_EXHAUST_X - x1) / 0.35);
        const alpha1 = Math.max(0, streamAlpha1 * boundFade1 * 0.88);

        // Point 2
        const u2 = (seg + 1) / SEGMENTS_PER_STREAM;
        const x2 = xVisStart + u2 * visSpan;
        const localU2 = Math.max(0, Math.min(1, (x2 - xTail) / s.length));
        const env2 = 0.45 + 0.55 * Math.sin(localU2 * Math.PI);

        const distFromHead2 = s.xHead - x2;
        const thetaY1_2 = s.spatialK1 * distFromHead2 + s.rippleFreq * t + s.phaseY1;
        const thetaY2_2 = s.spatialK2 * distFromHead2 + s.rippleFreq * 1.35 * t + s.phaseY2;
        const thetaZ_2 = s.spatialKz * distFromHead2 + s.rippleFreq * 0.9 * t + s.phaseZ;

        const y2 = s.baseY + env2 * (s.ampY1 * Math.sin(thetaY1_2) + s.ampY2 * Math.sin(thetaY2_2));
        const z2 = s.baseZ + env2 * (s.ampZ1 * Math.cos(thetaZ_2) + s.ampZ2 * Math.sin(thetaY1_2));

        const streamAlpha2 = Math.sin(Math.pow(localU2, 0.75) * Math.PI);
        const boundFade2 = Math.min(1, Math.min(x2 - RACK_INLET_X, RACK_EXHAUST_X - x2) / 0.35);
        const alpha2 = Math.max(0, streamAlpha2 * boundFade2 * 0.88);

        // Vertex 1 position & color
        positions[posPtr] = x1;
        positions[posPtr + 1] = y1;
        positions[posPtr + 2] = z1;
        posPtr += 3;

        colors[colPtr] = COLOR_R;
        colors[colPtr + 1] = COLOR_G;
        colors[colPtr + 2] = COLOR_B;
        colors[colPtr + 3] = alpha1;
        colPtr += 4;

        // Vertex 2 position & color
        positions[posPtr] = x2;
        positions[posPtr + 1] = y2;
        positions[posPtr + 2] = z2;
        posPtr += 3;

        colors[colPtr] = COLOR_R;
        colors[colPtr + 1] = COLOR_G;
        colors[colPtr + 2] = COLOR_B;
        colors[colPtr + 3] = alpha2;
        colPtr += 4;
      }
    }

    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;
  });

  return (
    <lineSegments ref={lineRef} geometry={geometry}>
      <lineBasicMaterial
        vertexColors
        transparent
        opacity={1.0}
        depthWrite={false}
        blending={THREE.NormalBlending}
      />
    </lineSegments>
  );
}
