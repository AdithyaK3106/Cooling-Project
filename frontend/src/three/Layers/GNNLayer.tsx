import { useMemo, useRef, useEffect } from 'react';
import { useTelemetry } from '../../services/telemetryApi';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Line } from '@react-three/drei';

// Maximum supported topology edges in the datacenter grid
const MAX_EDGES = 64;
// Resolution of the sinusoidal traveling heat wave along each edge
const SEGMENTS_PER_EDGE = 36;
// Number of traveling heat energy pulses along each edge
const PULSES_PER_EDGE = 2;
// Directional chevron segments per pulse: 2 wings + 1 spine + 3 trailing stream segments = 6
const CHEVRON_SEGMENTS_PER_PULSE = 6;
// Total segments per edge: 36 wave segments + (2 * 6) directional chevron segments = 48
const TOTAL_SEGMENTS_PER_EDGE = SEGMENTS_PER_EDGE + PULSES_PER_EDGE * CHEVRON_SEGMENTS_PER_PULSE;
const TOTAL_VERTICES = MAX_EDGES * TOTAL_SEGMENTS_PER_EDGE * 2;
const MAX_PULSES = MAX_EDGES * PULSES_PER_EDGE;

interface TopologyEdge {
  source: string;
  target: string;
  weight: number;
}

export function GNNLayer({ scene }: { scene: THREE.Object3D }) {
  const { data: telemetry } = useTelemetry();
  const lineRef = useRef<THREE.LineSegments>(null);
  const instancedMeshRef = useRef<THREE.InstancedMesh>(null);

  // 1. Map rack IDs to 3D world positions (exact existing rack positions)
  const positions = useMemo(() => {
    const map = new Map<string, THREE.Vector3>();
    if (!scene) return map;
    scene.traverse((child) => {
      if (child.userData && child.userData.rackId) {
        const pos = new THREE.Vector3();
        child.getWorldPosition(pos);
        map.set(child.userData.rackId, pos);
      }
    });
    return map;
  }, [scene]);

  // 2. Build edges matching exact existing graph topology and paths
  const edges = useMemo(() => {
    return (telemetry.topology as TopologyEdge[])
      .filter((edge: TopologyEdge) => positions.has(edge.source) && positions.has(edge.target))
      .map((edge: TopologyEdge, idx: number) => {
        const start = positions.get(edge.source)!;
        const end = positions.get(edge.target)!;

        // Exact existing quadratic bezier arch between connected racks
        const mid = start.clone().lerp(end, 0.5);
        mid.y += 5 * edge.weight; // Arch upwards based on risk weight

        const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
        const basePoints = curve.getPoints(24);
        const isHighRisk = edge.weight > 0.6;

        return {
          id: `${edge.source}->${edge.target}-${idx}`,
          source: edge.source,
          target: edge.target,
          weight: edge.weight,
          curve,
          basePoints,
          isHighRisk,
          color: isHighRisk ? '#ef4444' : '#f97316',
          rgb: isHighRisk
            ? { r: 239 / 255, g: 68 / 255, b: 68 / 255 }
            : { r: 249 / 255, g: 115 / 255, b: 22 / 255 },
          // Staggered organic phase offset so pulses ripple naturally across the network
          phaseOffset: (idx * 0.382) % (Math.PI * 2),
        };
      });
  }, [telemetry, positions]);

  // 3. Pre-allocate vertex buffers for silky-smooth 60 FPS animation (0 GC allocations per frame)
  const { geometry, posAttr, colAttr } = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    const pos = new Float32Array(TOTAL_VERTICES * 3);
    const col = new Float32Array(TOTAL_VERTICES * 4); // RGBA

    const pAttr = new THREE.BufferAttribute(pos, 3);
    const cAttr = new THREE.BufferAttribute(col, 4);

    geom.setAttribute('position', pAttr);
    geom.setAttribute('color', cAttr);

    return { geometry: geom, posAttr: pAttr, colAttr: cAttr };
  }, []);

  // Reusable vectors and dummy for instanced transforms
  const scratch = useMemo(() => ({
    up: new THREE.Vector3(0, 1, 0),
    tangent: new THREE.Vector3(),
    lateral: new THREE.Vector3(),
    normalV: new THREE.Vector3(),
    p1: new THREE.Vector3(),
    p2: new THREE.Vector3(),
    head: new THREE.Vector3(),
    leftWing: new THREE.Vector3(),
    rightWing: new THREE.Vector3(),
    spineTail: new THREE.Vector3(),
    trailPt: new THREE.Vector3(),
    dummy: new THREE.Object3D(),
    coreColor: new THREE.Color(),
  }), []);

  // Ensure instanced mesh updates count dynamically
  useEffect(() => {
    if (instancedMeshRef.current) {
      instancedMeshRef.current.count = Math.min(MAX_PULSES, edges.length * PULSES_PER_EDGE);
    }
  }, [edges.length]);

  // 4. Animation loop: update sinusoidal traveling heat waves and moving directional energy pulses
  useFrame((state) => {
    const t = state.clock.elapsedTime;
    const posArray = posAttr.array as Float32Array;
    const colArray = colAttr.array as Float32Array;

    let posPtr = 0;
    let colPtr = 0;
    let pulseIdx = 0;

    const { up, tangent, lateral, normalV, p1, p2, head, leftWing, rightWing, spineTail, trailPt, dummy, coreColor } = scratch;

    const numEdgesToRender = Math.min(edges.length, MAX_EDGES);

    for (let eIdx = 0; eIdx < numEdgesToRender; eIdx++) {
      const edge = edges[eIdx];
      const { curve, weight, rgb, isHighRisk, phaseOffset } = edge;

      // Traveling sinusoidal wave parameters: propagates along +u (from source rack to target rack)
      const waveFreq = 2.5 * Math.PI * 2; // ~2.5 heat-wave cycles along edge span
      const waveSpeed = 3.6;              // Fluid propagation speed (rad/s)
      const wavePhaseBase = phaseOffset - t * waveSpeed;

      // Restrained, thin enterprise-grade wave amplitudes
      const ampY = 0.12 + 0.08 * weight;
      const ampLat = 0.08 + 0.05 * weight;

      // Pulse traversal parameters: moving energy packets traveling from source -> destination
      const pulseSpeed = 0.38; // Traverses the entire edge span in ~2.6 seconds
      const pulseU = [
        (t * pulseSpeed + phaseOffset / (Math.PI * 2)) % 1.0,
        (t * pulseSpeed + 0.5 + phaseOffset / (Math.PI * 2)) % 1.0,
      ];

      // Helper to compute undulated point at parameter u along the curve
      const getUndulatedPoint = (u: number, out: THREE.Vector3) => {
        // Pinned envelope: exactly 0 at u = 0 (source) and u = 1 (destination)
        const env = Math.sin(Math.PI * u);
        curve.getPoint(u, out);

        curve.getTangent(u, tangent);
        lateral.crossVectors(tangent, up);
        if (lateral.lengthSq() < 0.0001) {
          lateral.set(1, 0, 0);
        } else {
          lateral.normalize();
        }
        normalV.crossVectors(lateral, tangent).normalize();

        const theta = waveFreq * u + wavePhaseBase;
        const dy = ampY * env * Math.sin(theta);
        const dLat = ampLat * env * Math.cos(theta * 1.15 + 0.3);

        out.addScaledVector(normalV, dy);
        out.addScaledVector(lateral, dLat);
      };

      // Helper to calculate pulse brightness glow at parameter u
      const getPulseGlow = (u: number) => {
        let maxGlow = 0;
        for (let p = 0; p < PULSES_PER_EDGE; p++) {
          let du = u - pulseU[p];
          if (du < -0.5) du += 1.0;
          if (du > 0.5) du -= 1.0;

          // Asymmetric comet falloff: crisp leading front, luminous trailing wake
          const falloff = du >= 0
            ? Math.exp(-(du * du) / (2 * 0.04 * 0.04))
            : Math.exp(-(du * du) / (2 * 0.12 * 0.12));

          if (falloff > maxGlow) maxGlow = falloff;
        }
        return maxGlow;
      };

      // ─── A. UNDULATING HEAT WAVE STREAMLINE ───
      for (let seg = 0; seg < SEGMENTS_PER_EDGE; seg++) {
        const u1 = seg / SEGMENTS_PER_EDGE;
        const u2 = (seg + 1) / SEGMENTS_PER_EDGE;

        getUndulatedPoint(u1, p1);
        getUndulatedPoint(u2, p2);

        const env1 = Math.sin(Math.PI * u1);
        const env2 = Math.sin(Math.PI * u2);

        const glow1 = getPulseGlow(u1);
        const glow2 = getPulseGlow(u2);

        // Alpha modulation: traveling heat pulses brighten the wave as they pass
        const alpha1 = Math.min(1.0, (0.40 + 0.60 * glow1) * (0.35 + 0.65 * env1) * (weight * 0.4 + 0.6));
        const alpha2 = Math.min(1.0, (0.40 + 0.60 * glow2) * (0.35 + 0.65 * env2) * (weight * 0.4 + 0.6));

        // Incandescent core brightening at pulse crests
        const r1 = rgb.r + (1.0 - rgb.r) * glow1 * 0.85;
        const g1 = rgb.g + (1.0 - rgb.g) * glow1 * 0.80;
        const b1 = rgb.b + (1.0 - rgb.b) * glow1 * 0.65;

        const r2 = rgb.r + (1.0 - rgb.r) * glow2 * 0.85;
        const g2 = rgb.g + (1.0 - rgb.g) * glow2 * 0.80;
        const b2 = rgb.b + (1.0 - rgb.b) * glow2 * 0.65;

        // Vertex 1
        posArray[posPtr++] = p1.x;
        posArray[posPtr++] = p1.y;
        posArray[posPtr++] = p1.z;
        colArray[colPtr++] = r1;
        colArray[colPtr++] = g1;
        colArray[colPtr++] = b1;
        colArray[colPtr++] = alpha1;

        // Vertex 2
        posArray[posPtr++] = p2.x;
        posArray[posPtr++] = p2.y;
        posArray[posPtr++] = p2.z;
        colArray[colPtr++] = r2;
        colArray[colPtr++] = g2;
        colArray[colPtr++] = b2;
        colArray[colPtr++] = alpha2;
      }

      // ─── B. DIRECTIONAL ARROW CHEVRONS & TRAILING PULSE STREAMS (Source -> Destination) ───
      for (let p = 0; p < PULSES_PER_EDGE; p++) {
        const u = pulseU[p];
        const env = Math.sin(Math.PI * u);

        if (env > 0.05) {
          getUndulatedPoint(u, head);
          curve.getTangent(u, tangent);
          lateral.crossVectors(tangent, up);
          if (lateral.lengthSq() < 0.0001) lateral.set(1, 0, 0);
          else lateral.normalize();

          // Arrow size scaled by path envelope
          const arrSize = (0.34 + 0.12 * weight) * env;

          // Tip of directional chevron (points toward destination rack)
          // Left wing tip (backswept)
          leftWing.copy(head)
            .addScaledVector(tangent, -arrSize * 1.0)
            .addScaledVector(lateral, arrSize * 0.55);

          // Right wing tip (backswept)
          rightWing.copy(head)
            .addScaledVector(tangent, -arrSize * 1.0)
            .addScaledVector(lateral, -arrSize * 0.55);

          // Spine tail
          spineTail.copy(head)
            .addScaledVector(tangent, -arrSize * 1.35);

          const headAlpha = Math.min(1.0, 0.98 * env);
          const wingAlpha = Math.min(1.0, 0.55 * env);

          // Head color: hot incandescent core
          const headR = 1.0;
          const headG = isHighRisk ? 0.90 : 0.96;
          const headB = isHighRisk ? 0.85 : 0.75;

          // 1. Chevron Wing Left (Head -> LeftWing)
          posArray[posPtr++] = head.x; posArray[posPtr++] = head.y; posArray[posPtr++] = head.z;
          colArray[colPtr++] = headR; colArray[colPtr++] = headG; colArray[colPtr++] = headB; colArray[colPtr++] = headAlpha;
          posArray[posPtr++] = leftWing.x; posArray[posPtr++] = leftWing.y; posArray[posPtr++] = leftWing.z;
          colArray[colPtr++] = rgb.r; colArray[colPtr++] = rgb.g; colArray[colPtr++] = rgb.b; colArray[colPtr++] = wingAlpha;

          // 2. Chevron Wing Right (Head -> RightWing)
          posArray[posPtr++] = head.x; posArray[posPtr++] = head.y; posArray[posPtr++] = head.z;
          colArray[colPtr++] = headR; colArray[colPtr++] = headG; colArray[colPtr++] = headB; colArray[colPtr++] = headAlpha;
          posArray[posPtr++] = rightWing.x; posArray[posPtr++] = rightWing.y; posArray[posPtr++] = rightWing.z;
          colArray[colPtr++] = rgb.r; colArray[colPtr++] = rgb.g; colArray[colPtr++] = rgb.b; colArray[colPtr++] = wingAlpha;

          // 3. Chevron Spine (Head -> SpineTail)
          posArray[posPtr++] = head.x; posArray[posPtr++] = head.y; posArray[posPtr++] = head.z;
          colArray[colPtr++] = headR; colArray[colPtr++] = headG; colArray[colPtr++] = headB; colArray[colPtr++] = headAlpha;
          posArray[posPtr++] = spineTail.x; posArray[posPtr++] = spineTail.y; posArray[posPtr++] = spineTail.z;
          colArray[colPtr++] = rgb.r; colArray[colPtr++] = rgb.g; colArray[colPtr++] = rgb.b; colArray[colPtr++] = wingAlpha;

          // 4, 5, 6. Trailing stream line segments behind the pulse
          const trailDeltas = [0.035, 0.07, 0.105, 0.14];
          for (let s = 0; s < 3; s++) {
            const ut1 = Math.max(0, u - trailDeltas[s]);
            const ut2 = Math.max(0, u - trailDeltas[s + 1]);
            getUndulatedPoint(ut1, trailPt);
            posArray[posPtr++] = trailPt.x; posArray[posPtr++] = trailPt.y; posArray[posPtr++] = trailPt.z;
            colArray[colPtr++] = rgb.r; colArray[colPtr++] = rgb.g; colArray[colPtr++] = rgb.b;
            colArray[colPtr++] = Math.max(0, (0.75 - s * 0.20) * env);

            getUndulatedPoint(ut2, trailPt);
            posArray[posPtr++] = trailPt.x; posArray[posPtr++] = trailPt.y; posArray[posPtr++] = trailPt.z;
            colArray[colPtr++] = rgb.r; colArray[colPtr++] = rgb.g; colArray[colPtr++] = rgb.b;
            colArray[colPtr++] = Math.max(0, (0.55 - s * 0.20) * env);
          }

          // Instanced glowing core bead at pulse head
          if (instancedMeshRef.current && pulseIdx < MAX_PULSES) {
            dummy.position.copy(head);
            const scale = Math.max(0.001, (0.15 + 0.07 * weight) * env);
            dummy.scale.set(scale, scale, scale);
            dummy.updateMatrix();
            instancedMeshRef.current.setMatrixAt(pulseIdx, dummy.matrix);

            coreColor.setRGB(headR, headG, headB);
            instancedMeshRef.current.setColorAt(pulseIdx, coreColor);
            pulseIdx++;
          }
        } else {
          // Collapse unused chevron segments when pulse is near rack endpoints
          for (let k = 0; k < CHEVRON_SEGMENTS_PER_PULSE * 2; k++) {
            posArray[posPtr++] = 0; posArray[posPtr++] = 0; posArray[posPtr++] = 0;
            colArray[colPtr++] = 0; colArray[colPtr++] = 0; colArray[colPtr++] = 0; colArray[colPtr++] = 0;
          }

          // Collapse instanced bead
          if (instancedMeshRef.current && pulseIdx < MAX_PULSES) {
            dummy.position.set(0, -999, 0);
            dummy.scale.set(0, 0, 0);
            dummy.updateMatrix();
            instancedMeshRef.current.setMatrixAt(pulseIdx, dummy.matrix);
            pulseIdx++;
          }
        }
      }
    }

    // Collapse any remaining unused vertices
    while (posPtr < posArray.length) {
      posArray[posPtr++] = 0;
      colArray[colPtr++] = 0;
    }

    // Hide any remaining instances
    if (instancedMeshRef.current) {
      while (pulseIdx < instancedMeshRef.current.count) {
        dummy.position.set(0, -999, 0);
        dummy.scale.set(0, 0, 0);
        dummy.updateMatrix();
        instancedMeshRef.current.setMatrixAt(pulseIdx, dummy.matrix);
        pulseIdx++;
      }
      instancedMeshRef.current.instanceMatrix.needsUpdate = true;
      if (instancedMeshRef.current.instanceColor) {
        instancedMeshRef.current.instanceColor.needsUpdate = true;
      }
    }

    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
  });

  return (
      {/* 1. Base Subtle Guide Arch (Clean Wireframe Baseline) */}
      {edges.map((edge) => (
        <Line
          key={`base-${edge.id}`}
          points={edge.basePoints}
          color={edge.color}
          lineWidth={edge.weight * 2.2 + 0.8}
          transparent
          opacity={edge.weight * 0.45 + 0.25}
        />
      ))}

      {/* 2. Dynamic Flowing Heat Wave Segments & Directional Chevrons */}
      <lineSegments ref={lineRef} geometry={geometry}>
        <lineBasicMaterial
          vertexColors
          transparent
          opacity={1.0}
          depthWrite={false}
          blending={THREE.NormalBlending}
        />
      </lineSegments>

      {/* 3. Incandescent Pulse Core Beads */}
      <instancedMesh
        ref={instancedMeshRef}
        args={[undefined, undefined, MAX_PULSES]}
        frustumCulled={false}
      >
        <sphereGeometry args={[1.0, 8, 8]} />
        <meshBasicMaterial
          transparent
          opacity={0.92}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </instancedMesh>
    </>
  );
}

