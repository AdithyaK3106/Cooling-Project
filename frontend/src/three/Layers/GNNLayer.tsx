import { useMemo, useRef, useEffect } from 'react';
import { useTelemetry } from '../../services/telemetryApi';
import { getSimulatedTelemetry } from '../../services/simulation';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { Line } from '@react-three/drei';

// Maximum supported topology edges in the datacenter grid (40 edges in 5x5 grid)
const MAX_EDGES = 64;
// Number of traveling directional chevron pulses along each straight connection line
const PULSES_PER_EDGE = 2;
// Segments per chevron: tip->left wing (1), tip->right wing (2), tip->spine tail (3)
const CHEVRON_SEGMENTS = 3;
const TOTAL_SEGMENTS_PER_EDGE = PULSES_PER_EDGE * CHEVRON_SEGMENTS;
const TOTAL_VERTICES = MAX_EDGES * TOTAL_SEGMENTS_PER_EDGE * 2;

// Existing risk color thresholds used across the application (RacksTab, OverviewTab, KeyPerformancePanel):
// - HIGH / CRITICAL: risk_score > 0.55 -> Red
// - MEDIUM / MODERATE: risk_score >= 0.35 && <= 0.55 -> Amber/Yellow
// - LOW: risk_score < 0.35 -> Cool Blue/Cyan
const RISK_RACK_COLORS = {
  low: new THREE.Color('#06b6d4'),     // Cool cyan / sky blue (restrained)
  medium: new THREE.Color('#f59e0b'),  // Amber / warm yellow
  high: new THREE.Color('#ef4444'),    // Red
};

interface TopologyEdge {
  source: string;
  target: string;
  weight: number;
}

// Helper to extract integer rack number (e.g. 'A001' -> 1, 'A01' -> 1, 'Rack 12' -> 12)
const getRackNumber = (id: string): number | null => {
  if (!id) return null;
  const match = id.match(/\d+/);
  return match ? parseInt(match[0], 10) : null;
};

export function GNNLayer({ scene }: { scene: THREE.Object3D }) {
  const { data: telemetry } = useTelemetry();
  const lineRef = useRef<THREE.LineSegments>(null);

  // 1. Map rack IDs and rack numbers to exact 3D world positions
  const { positions, rackNumMap } = useMemo(() => {
    const posMap = new Map<string, THREE.Vector3>();
    const numMap = new Map<number, THREE.Vector3>();
    if (!scene) return { positions: posMap, rackNumMap: numMap };

    scene.traverse((child) => {
      if (child.userData && child.userData.rackId) {
        const pos = new THREE.Vector3();
        child.getWorldPosition(pos);
        const rawId = child.userData.rackId;
        posMap.set(rawId, pos);

        const num = getRackNumber(rawId);
        if (num !== null) {
          numMap.set(num, pos);
          posMap.set(`A0${num}`, pos);
          posMap.set(`A00${num}`, pos);
          posMap.set(`A0${num < 10 ? '0' + num : num}`, pos);
          posMap.set(`A${num}`, pos);
          posMap.set(`Rack ${num}`, pos);
          posMap.set(`Rack_${num}`, pos);
        }
      }
    });
    return { positions: posMap, rackNumMap: numMap };
  }, [scene]);

  // Helper to resolve 3D position for any rack identifier variation
  const resolvePosition = (id: string): THREE.Vector3 | undefined => {
    if (positions.has(id)) return positions.get(id);
    const num = getRackNumber(id);
    if (num !== null && rackNumMap.has(num)) return rackNumMap.get(num);
    return undefined;
  };

  // 1b. Cache rack meshes and clone materials once for dynamic risk color mapping
  const rackMeshesMap = useMemo(() => {
    const map = new Map<string, THREE.Mesh[]>();
    if (!scene) return map;

    scene.traverse((child) => {
      if (child.userData && child.userData.rackId) {
        const accentMeshes: THREE.Mesh[] = [];
        const fallbackMeshes: THREE.Mesh[] = [];
        child.traverse((mesh: any) => {
          if (mesh.isMesh && mesh.material) {
            if (mesh.userData?.isRiskAccent || (mesh.material as any)?.__isRiskAccentMaterial) {
              accentMeshes.push(mesh);
            } else {
              fallbackMeshes.push(mesh);
            }
          }
        });

        // Use accent meshes if detailed enterprise cabinet is present, maintaining dark metallic chassis
        if (accentMeshes.length > 0) {
          map.set(child.userData.rackId, accentMeshes);
        } else {
          fallbackMeshes.forEach((mesh) => {
            if (!mesh.userData.__gnnRiskMaterialCloned && !mesh.userData.uniqueMaterial) {
              mesh.material = Array.isArray(mesh.material)
                ? mesh.material.map((m: any) => m.clone())
                : mesh.material.clone();
              mesh.userData.__gnnRiskMaterialCloned = true;
            }
          });
          map.set(child.userData.rackId, fallbackMeshes);
        }
      }
    });

    return map;
  }, [scene]);

  // Reset rack emissives on unmount (e.g., when switching from RISK to AIRFLOW)
  useEffect(() => {
    return () => {
      rackMeshesMap.forEach((meshes) => {
        meshes.forEach((mesh) => {
          const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
          mats.forEach((mat: any) => {
            if (mat.emissive !== undefined) {
              mat.emissive.set(0x000000);
              mat.emissiveIntensity = 0;
              mat.needsUpdate = true;
            }
          });
        });
      });
    };
  }, [rackMeshesMap]);

  // 2. Build straight edges matching exact existing graph topology and paths, connecting EVERY rack
  const edges = useMemo(() => {
    // Obtain the existing network topology from telemetry or simulated fleet
    const sourceTopology = (telemetry?.topology && Array.isArray(telemetry.topology) && telemetry.topology.length > 0)
      ? (telemetry.topology as TopologyEdge[])
      : (getSimulatedTelemetry()?.topology as TopologyEdge[] || []);

    if (!sourceTopology || sourceTopology.length === 0) return [];

    return sourceTopology
      .filter((edge: TopologyEdge) => {
        const startPos = resolvePosition(edge.source);
        const endPos = resolvePosition(edge.target);
        return !!startPos && !!endPos;
      })
      .map((edge: TopologyEdge, idx: number) => {
        const startRaw = resolvePosition(edge.source)!;
        const endRaw = resolvePosition(edge.target)!;
        const pStart = new THREE.Vector3(startRaw.x, 0.65, startRaw.z);
        const pEnd = new THREE.Vector3(endRaw.x, 0.65, endRaw.z);
        const dir = new THREE.Vector3().subVectors(pEnd, pStart).normalize();
        const dist = pStart.distanceTo(pEnd);
        let lateral = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(0, 1, 0)).normalize();
        if (lateral.lengthSq() < 0.0001) lateral.set(0, 0, 1);

        const edgeSourceNum = getRackNumber(edge.source);
        const sourceRack = telemetry?.racks?.find((r: any) => {
          if (r.id === edge.source) return true;
          const rNum = getRackNumber(r.id);
          return rNum !== null && edgeSourceNum !== null && rNum === edgeSourceNum;
        });
        const sourceRisk = sourceRack?.risk_score ?? 0;
        const effectiveHeat = Math.max(edge.weight, sourceRisk * 0.88);

        // Heat flow color tiers matching user specification:
        // HIGH HEAT FLOW (> 0.55) -> deep professional red
        // MODERATE HEAT FLOW (0.35 - 0.55) -> amber/orange
        // LOW HEAT FLOW (< 0.35) -> soft cyan/blue
        let color = '#0284c7'; // Soft cyan/blue (< 0.35)
        let rgb = { r: 2 / 255, g: 132 / 255, b: 199 / 255 };
        let lineWidth = 1.4;

        if (effectiveHeat > 0.55) {
          color = '#b91c1c'; // Deep professional red (> 0.55)
          rgb = { r: 185 / 255, g: 28 / 255, b: 28 / 255 };
          lineWidth = 2.0;
        } else if (effectiveHeat >= 0.35) {
          color = '#d97706'; // Warm amber/orange (0.35 - 0.55)
          rgb = { r: 217 / 255, g: 119 / 255, b: 6 / 255 };
          lineWidth = 1.6;
        }

        return {
          id: `${edge.source}->${edge.target}-${idx}`,
          source: edge.source,
          target: edge.target,
          weight: edge.weight,
          effectiveHeat,
          pStart,
          pEnd,
          basePoints: [pStart, pEnd],
          dir,
          dist,
          lateral,
          color,
          rgb,
          lineWidth,
          phaseOffset: (idx * 0.382) % (Math.PI * 2),
        };
      });
  }, [telemetry, positions, rackNumMap]);

  // 3. Pre-allocated vertex buffers for directional chevron animation
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

  // Reusable scratch vectors
  const scratch = useMemo(() => ({
    head: new THREE.Vector3(),
    leftWing: new THREE.Vector3(),
    rightWing: new THREE.Vector3(),
    spineTail: new THREE.Vector3(),
  }), []);

  // 4. Animation loop: update small directional chevrons & dynamic rack risk colors
  useFrame((state, delta) => {
    const t = state.clock.elapsedTime;
    const posArray = posAttr.array as Float32Array;
    const colArray = colAttr.array as Float32Array;

    let posPtr = 0;
    let colPtr = 0;

    const { head, leftWing, rightWing, spineTail } = scratch;
    const numEdgesToRender = Math.min(edges.length, MAX_EDGES);

    // Directional chevron animation traveling: SOURCE RACK -> heat influence -> CONNECTED RACK
    const pulseSpeed = 0.32; // Smooth, continuous, restrained traversal (~3.1 seconds across line)

    for (let eIdx = 0; eIdx < numEdgesToRender; eIdx++) {
      const edge = edges[eIdx];
      const { pStart, pEnd, dir, lateral, rgb, phaseOffset } = edge;

      for (let p = 0; p < PULSES_PER_EDGE; p++) {
        // Continuous travel from 0.0 (source) to 1.0 (destination)
        const u = (t * pulseSpeed + phaseOffset / (Math.PI * 2) + p * 0.5) % 1.0;
        // Pinned envelope: fades in smoothly as it leaves source, fades out near target
        const env = Math.sin(Math.PI * u);

        if (env > 0.05) {
          // Current chevron position along the straight line
          head.lerpVectors(pStart, pEnd, u);

          // Proportional chevron geometry (compact, elegant, clearly directional)
          const chevronLength = 0.22;
          const chevronWidth = 0.12;

          leftWing.copy(head)
            .addScaledVector(dir, -chevronLength)
            .addScaledVector(lateral, chevronWidth);

          rightWing.copy(head)
            .addScaledVector(dir, -chevronLength)
            .addScaledVector(lateral, -chevronWidth);

          spineTail.copy(head)
            .addScaledVector(dir, -chevronLength * 0.65);

          const alphaHead = Math.min(1.0, 0.95 * env);
          const alphaWing = Math.min(1.0, 0.70 * env);

          // Core head brightness: refined bright core with restrained glow matching its line family
          const headR = Math.min(1.0, rgb.r * 1.30);
          const headG = Math.min(1.0, rgb.g * 1.30);
          const headB = Math.min(1.0, rgb.b * 1.30);

          // Segment 1: Tip -> Left Wing
          posArray[posPtr++] = head.x; posArray[posPtr++] = head.y; posArray[posPtr++] = head.z;
          colArray[colPtr++] = headR; colArray[colPtr++] = headG; colArray[colPtr++] = headB; colArray[colPtr++] = alphaHead;
          posArray[posPtr++] = leftWing.x; posArray[posPtr++] = leftWing.y; posArray[posPtr++] = leftWing.z;
          colArray[colPtr++] = rgb.r; colArray[colPtr++] = rgb.g; colArray[colPtr++] = rgb.b; colArray[colPtr++] = alphaWing;

          // Segment 2: Tip -> Right Wing
          posArray[posPtr++] = head.x; posArray[posPtr++] = head.y; posArray[posPtr++] = head.z;
          colArray[colPtr++] = headR; colArray[colPtr++] = headG; colArray[colPtr++] = headB; colArray[colPtr++] = alphaHead;
          posArray[posPtr++] = rightWing.x; posArray[posPtr++] = rightWing.y; posArray[posPtr++] = rightWing.z;
          colArray[colPtr++] = rgb.r; colArray[colPtr++] = rgb.g; colArray[colPtr++] = rgb.b; colArray[colPtr++] = alphaWing;

          // Segment 3: Tip -> Central Spine Tail
          posArray[posPtr++] = head.x; posArray[posPtr++] = head.y; posArray[posPtr++] = head.z;
          colArray[colPtr++] = headR; colArray[colPtr++] = headG; colArray[colPtr++] = headB; colArray[colPtr++] = alphaHead;
          posArray[posPtr++] = spineTail.x; posArray[posPtr++] = spineTail.y; posArray[posPtr++] = spineTail.z;
          colArray[colPtr++] = rgb.r; colArray[colPtr++] = rgb.g; colArray[colPtr++] = rgb.b; colArray[colPtr++] = alphaWing;
        }
      }
    }

    // Zero out unused vertices
    while (posPtr < posArray.length) {
      posArray[posPtr++] = 0;
    }
    while (colPtr < colArray.length) {
      colArray[colPtr++] = 0;
    }

    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;

    // Smooth lerp of rack colors based on their existing live risk scores
    const lerpFactor = Math.min(1.0, delta * 4.0);

    rackMeshesMap.forEach((meshes, rackId) => {
      const rackNum = getRackNumber(rackId);
      const rack = telemetry?.racks?.find((r: any) => {
        if (r.id === rackId) return true;
        const rNum = getRackNumber(r.id);
        return rNum !== null && rackNum !== null && rNum === rackNum;
      });
      const risk = rack?.risk_score ?? 0;

      // Reuses existing fleet classification (RacksTab, OverviewTab, KeyPerformancePanel):
      // High/Critical: risk > 0.55 -> Red
      // Medium/Moderate: risk >= 0.35 && <= 0.55 -> Amber/Yellow
      // Low: risk < 0.35 -> Cool Blue/Cyan
      let targetColor = RISK_RACK_COLORS.low;
      let targetIntensity = 0.60;

      if (risk > 0.55) {
        targetColor = RISK_RACK_COLORS.high;
        targetIntensity = 0.90;
      } else if (risk >= 0.35) {
        targetColor = RISK_RACK_COLORS.medium;
        targetIntensity = 0.75;
      }

      for (let m = 0; m < meshes.length; m++) {
        const mesh = meshes[m];
        const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        for (let j = 0; j < mats.length; j++) {
          const mat = mats[j] as any;
          if (mat.emissive !== undefined) {
            mat.emissive.lerp(targetColor, lerpFactor);
            mat.emissiveIntensity = THREE.MathUtils.lerp(
              mat.emissiveIntensity !== undefined ? mat.emissiveIntensity : 0,
              targetIntensity,
              lerpFactor
            );
            mat.needsUpdate = true;
          }
        }
      }
    });
  });

  return (
    <>
      {/* 1. Straight, clean heat-flow guide lines between connected racks */}
      {edges.map((edge) => (
        <Line
          key={`base-${edge.id}`}
          points={edge.basePoints}
          color={edge.color}
          lineWidth={edge.lineWidth}
          transparent
          opacity={0.90}
          raycast={() => null}
        />
      ))}

      {/* 2. Restrained directional chevrons traveling smoothly along the straight lines */}
      <lineSegments ref={lineRef} geometry={geometry} raycast={() => null}>
        <lineBasicMaterial
          vertexColors
          transparent
          opacity={1.0}
          depthWrite={false}
          blending={THREE.NormalBlending}
        />
      </lineSegments>
    </>
  );
}
