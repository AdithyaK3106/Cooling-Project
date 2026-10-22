import React, { Suspense } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Environment, useGLTF } from '@react-three/drei';
import { CameraController } from '../Camera';

import { useUiStore } from '../../stores/uiStore';

import { ThermalLayer } from '../Layers/ThermalLayer';
import { GNNLayer } from '../Layers/GNNLayer';
import { AirflowLayer } from '../Layers/AirflowLayer';
import { StatsLayer } from '../Layers/StatsLayer';
import { Html } from '@react-three/drei';
import { useQueryClient } from '@tanstack/react-query';
import { useTelemetry, TELEMETRY_QUERY_KEY } from '../../services/telemetryApi';
import { tickSimulation, getSimulatedTelemetry, setRackCooling } from '../../services/simulation';
import { useSetRackCooling } from '../../services/controlApi';
import { createDetailedServerRack } from '../models/ServerRackCabinet';
import { PerimeterWall } from '../models/PerimeterWall';
import { DataCenterFloor } from '../models/DataCenterFloor';
import * as THREE from 'three';

function Model() {
  const { scene } = useGLTF('/models/room_server.glb');
  const { selectedRackId, setSelectedRackId, hoveredRackId, setHoveredRackId, activeLayer } = useUiStore();
  const { data: telemetry } = useTelemetry();
  const queryClient = useQueryClient();
  const setRackCoolingMutation = useSetRackCooling();
  const [actionTick, setActionTick] = React.useState(0);
  const [simTick, setSimTick] = React.useState(0);

  const isOverCardRef = React.useRef(false);
  const hoverTimeoutRef = React.useRef<any>(null);

  const clearHoverTimeout = () => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
      hoverTimeoutRef.current = null;
    }
  };

  const handleDeployCooling = (targetRackId: string) => {
    setRackCooling(targetRackId, true);
    setRackCoolingMutation.mutate({ rackId: targetRackId, status: 'predictive intervention' });
    queryClient.setQueryData(TELEMETRY_QUERY_KEY, (oldData: any) => {
      if (!oldData || !oldData.racks) return oldData;
      const numMatch = targetRackId.match(/\d+/);
      const rackNum = numMatch ? parseInt(numMatch[0], 10) : -1;
      return {
        ...oldData,
        racks: oldData.racks.map((item: any) => {
          const match = item.id === targetRackId || 
                        item.id === `A0${rackNum}` || 
                        item.id === `A${rackNum}` ||
                        (rackNum > 0 && parseInt(item.id.replace(/\D+/g, ''), 10) === rackNum);
          if (!match) return item;
          return {
            ...item,
            coolingActive: true,
            overrideEnabled: true,
            cooling: { ...item.cooling, status: 'predictive intervention' }
          };
        })
      };
    });
    queryClient.invalidateQueries({ queryKey: TELEMETRY_QUERY_KEY });
    setActionTick((t) => t + 1);
  };

  const handleDisengage = (targetRackId: string) => {
    setRackCooling(targetRackId, false);
    setRackCoolingMutation.mutate({ rackId: targetRackId, status: 'normal' });
    queryClient.setQueryData(TELEMETRY_QUERY_KEY, (oldData: any) => {
      if (!oldData || !oldData.racks) return oldData;
      const numMatch = targetRackId.match(/\d+/);
      const rackNum = numMatch ? parseInt(numMatch[0], 10) : -1;
      return {
        ...oldData,
        racks: oldData.racks.map((item: any) => {
          const match = item.id === targetRackId || 
                        item.id === `A0${rackNum}` || 
                        item.id === `A${rackNum}` ||
                        (rackNum > 0 && parseInt(item.id.replace(/\D+/g, ''), 10) === rackNum);
          if (!match) return item;
          return {
            ...item,
            coolingActive: false,
            overrideEnabled: false,
            cooling: { ...item.cooling, status: 'normal' }
          };
        })
      };
    });
    queryClient.invalidateQueries({ queryKey: TELEMETRY_QUERY_KEY });
    setActionTick((t) => t + 1);
  };

  React.useEffect(() => {
    (window as any).__deployCoolingOnHovered = () => {
      if (hoveredRackId) handleDeployCooling(hoveredRackId);
    };
    (window as any).__disengageOnHovered = () => {
      if (hoveredRackId) handleDisengage(hoveredRackId);
    };
    (window as any).__scene = scene;
    (window as any).__openRack = (id: string) => {
      setSelectedRackId(id);
      setHoveredRackId(id);
    };
    (window as any).__getHoveredRack = () => hoveredRackId;
    (window as any).__getSelectedRack = () => selectedRackId;
  }, [hoveredRackId, selectedRackId, scene]);

  const [isReady, setIsReady] = React.useState(false);

  React.useEffect(() => {
    const nodesToRemove: THREE.Object3D[] = [];

    scene.traverse((child: any) => {
      const lowerName = child.name.toLowerCase();
      const parentLowerName = child.parent?.name?.toLowerCase() || '';

      // Hide all old room geometry from the GLB (floors, walls, ceilings, columns, doors, glass)
      // leaving only our custom high-tech DataCenterFloor and PerimeterWall
      if (
        lowerName.includes('bottom_plane') || lowerName.includes('floor') ||
        lowerName.includes('interior') || lowerName.includes('exterior') || lowerName.includes('top_plane') ||
        lowerName.includes('roof') || lowerName.includes('ceiling') || lowerName.includes('top plane') ||
        lowerName.includes('wall') || parentLowerName.includes('wall') ||
        lowerName.includes('glass') || parentLowerName.includes('glass') ||
        lowerName.includes('door') || parentLowerName.includes('door') ||
        lowerName.includes('window') || parentLowerName.includes('window') ||
        lowerName.includes('column') || parentLowerName.includes('column') ||
        lowerName.includes('plant') || parentLowerName.includes('plant') ||
        lowerName.includes('flowerpot') || parentLowerName.includes('flowerpot') ||
        lowerName.includes('metal') || parentLowerName.includes('metal')
      ) {
        child.visible = false;
      }

      let isPartOfRack = false;
      let node = child;
      while (node) {
        if (node.name.match(/Rack/i)) {
          isPartOfRack = true;
          break;
        }
        node = node.parent;
      }

      if (!isPartOfRack) {
        child.raycast = () => null;
      }

      const match = child.name.match(/^Rack\s*(\d+)$/i) || child.name.match(/^Rack_(\d+)$/i);
      if (match) {
        const num = parseInt(match[1], 10);
        
        if (num > 25) {
          nodesToRemove.push(child);
          return;
        }

        const rackId = `A0${num}`;
        child.userData = { rackId };

        const index = num - 1;
        const row = Math.floor(index / 5);
        const col = index % 5;
        
        const spacingX = 3.5;
        const spacingZ = 5.0;
        
        child.position.set(
          (col - 2) * spacingX,
          0,
          (row - 2) * spacingZ
        );
        child.rotation.set(0, 0, 0);

        child.children.forEach((c: any) => {
          if (!c.userData?.isDetailedCabinet) {
            c.visible = false;
          }
        });

        if (!child.children.some((c: any) => c.userData?.isDetailedCabinet)) {
          const cabinet = createDetailedServerRack(rackId);
          child.add(cabinet);
        }
      }
    });

    nodesToRemove.forEach(node => node.removeFromParent());
    setIsReady(true);
  }, [scene]);

  const hoveredRackPos = React.useMemo(() => {
    if (!hoveredRackId) return null;
    let pos = new THREE.Vector3();
    scene.traverse((child) => {
      if (child.userData.rackId === hoveredRackId) {
        child.getWorldPosition(pos);
        pos.y += 1.3;
      }
    });
    return pos;
  }, [hoveredRackId, scene]);

  React.useEffect(() => {
    tickSimulation();
    const interval = setInterval(() => {
      tickSimulation();
      setSimTick((t) => t + 1);
    }, 200);
    return () => clearInterval(interval);
  }, []);

  const hoveredRackData = React.useMemo(() => {
    if (!hoveredRackId) return null;
    try {
      const sim = getSimulatedTelemetry();
      const numMatch = hoveredRackId.match(/\d+/);
      const rackNum = numMatch ? parseInt(numMatch[0], 10) : -1;
      const matchRack = (item: any) => 
        item.id === hoveredRackId || 
        item.id === `A0${rackNum}` || 
        item.id === `A${rackNum}` ||
        (rackNum > 0 && parseInt(item.id.replace(/\D+/g, ''), 10) === rackNum);

      const simRack = sim?.racks?.find(matchRack);
      const fromTelem = telemetry?.racks?.find(matchRack);
      if (simRack) {
        return {
          ...(fromTelem || {}),
          ...simRack,
          telemetry: {
            ...(fromTelem?.telemetry || {}),
            ...simRack.telemetry
          },
          ai_insights: {
            ...(fromTelem?.ai_insights || {}),
            ...simRack.ai_insights
          },
          risk_score: simRack.risk_score,
          overrideEnabled: simRack.overrideEnabled,
          coolingActive: simRack.coolingActive,
          cooling: { status: (simRack.coolingActive || simRack.overrideEnabled) ? 'predictive intervention' : (fromTelem?.cooling?.status || 'normal') }
        };
      }
      return fromTelem || null;
    } catch {
      return null;
    }
  }, [telemetry, hoveredRackId, actionTick, simTick]);

  return (
    <group>
      <primitive 
        object={scene} 
        onDoubleClick={(e: any) => {
          e.stopPropagation();
          let node = e.object;
          while (node && !node.userData?.rackId && !(node.name.match(/^Rack\s*\d+$/i) || node.name.match(/^Rack_\d+$/i)) && node.parent) {
            node = node.parent;
          }
          const rackId = node?.userData?.rackId || (node?.name?.match(/^Rack\s*(\d+)$/i) ? `A0${parseInt(node.name.match(/^Rack\s*(\d+)$/i)[1], 10)}` : null);
          if (rackId) {
            clearHoverTimeout();
            isOverCardRef.current = false;
            setSelectedRackId(rackId);
            setHoveredRackId(rackId);
          }
        }}
        onClick={(e: any) => {
          e.stopPropagation();
          // Single-click does NOT open the rack information panel
        }}
        onPointerOver={(e: any) => {
          e.stopPropagation();
          let node = e.object;
          while (node && !node.userData?.rackId && !(node.name.match(/^Rack\s*\d+$/i) || node.name.match(/^Rack_\d+$/i)) && node.parent) {
            node = node.parent;
          }
          if (node && (node.userData?.rackId || node.name.match(/^Rack\s*\d+$/i) || node.name.match(/^Rack_\d+$/i))) {
            document.body.style.cursor = 'pointer';
          }
        }}
        onPointerOut={(e: any) => {
          e.stopPropagation();
          document.body.style.cursor = 'default';
        }}
        onPointerMissed={() => {
          setSelectedRackId(null);
          setHoveredRackId(null);
        }}
      />
      
      {hoveredRackPos && hoveredRackData && (() => {
        const r = hoveredRackData;
        const numMatch = (r.id || hoveredRackId || '').match(/\d+/);
        const rackNum = numMatch ? parseInt(numMatch[0], 10) : 1;
        const displayRackId = `RACK-A${rackNum < 10 ? `0${rackNum}` : rackNum}`;
        const nodeIndex = rackNum - 1;
        const zone = r.ai_insights?.zone || ['A', 'B', 'C', 'D', 'E'][Math.floor(nodeIndex / 5)] || 'A';
        const isCooling = r.cooling?.status === 'predictive intervention' || r.coolingActive || r.overrideEnabled;
        const riskScore = typeof r.risk_score === 'number' ? r.risk_score : (r.riskScore ?? 0.15);
        const riskPct = Math.round(riskScore * 100);

        const isCritical = riskScore >= 0.70;
        const isHigh = riskScore >= 0.55 && !isCritical;
        const isMedium = riskScore >= 0.35 && !isHigh && !isCritical;

        const riskLabel = isCritical ? 'CRITICAL' : isHigh ? 'HIGH' : isMedium ? 'MED' : 'LOW';
        const riskBadgeClass = isCritical 
          ? 'bg-red-500/20 text-red-400 border-red-500/40' 
          : isHigh 
          ? 'bg-orange-500/20 text-orange-400 border-orange-500/40' 
          : isMedium 
          ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/40' 
          : 'bg-green-500/20 text-green-400 border-green-500/40';

        const riskColor = isCritical ? 'text-red-400' : isHigh ? 'text-orange-400' : isMedium ? 'text-yellow-400' : 'text-green-400';

        const cpu = r.telemetry?.cpu_util ?? r.cpu ?? 0;
        const gpu = r.telemetry?.gpu_util ?? r.gpu ?? 0;
        const temp = r.telemetry?.cpu_temp ?? (32 + riskScore * 48);

        const xgbPred = r.ai_insights?.xgb_pred ?? r.xgbPred ?? Math.min(0.99, riskScore * 0.96);
        const gnnEmbed = r.ai_insights?.gnn_embed ?? r.gnnEmbed ?? Math.min(0.99, riskScore * 0.88);
        const neighborInfluence = Math.round(gnnEmbed * 12);
        const isRising = riskScore >= 0.45;
        const riskTrend = isRising ? '↑ Increasing' : '↓ Decreasing';

        let recBadge = 'Normal Operation';
        let recBadgeColor = 'text-green-400';
        let recText = `Thermal parameters are nominal. No active intervention required for ${displayRackId}.`;

        if (isCooling) {
          recBadge = 'Maintain Cooling';
          recBadgeColor = 'text-cyan-400';
          recText = `Maintain active cooling on ${displayRackId}. Thermal risk is managed with +${neighborInfluence}% neighbor influence.`;
        } else if (riskScore > 0.58) {
          recBadge = 'Proactive Intervention';
          recBadgeColor = 'text-red-400';
          recText = `${displayRackId} has crossed the 58% cooling threshold. Automated intervention engaged.`;
        } else if (riskScore > 0.45) {
          recBadge = 'Elevated Monitoring';
          recBadgeColor = 'text-yellow-400';
          recText = `Proactive monitoring for ${displayRackId}. Workload intensity is elevating thermal risk.`;
        }

        const isRightSide = hoveredRackPos.x > 0;
        const transformOffset = isRightSide ? 'translate3d(-105%, -40%, 0)' : 'translate3d(20px, -40%, 0)';

        return (
          <Html 
            position={hoveredRackPos} 
            center 
            style={{ 
              pointerEvents: 'auto', 
              zIndex: 1000,
              transform: transformOffset
            }}
          >
            <div 
              onMouseEnter={() => {
                isOverCardRef.current = true;
                clearHoverTimeout();
              }}
              onMouseLeave={() => {
                isOverCardRef.current = false;
              }}
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => e.stopPropagation()}
              onDoubleClick={(e) => e.stopPropagation()}
              className="w-72 rounded-xl border border-white/15 bg-[#0B0E14]/95 p-3.5 backdrop-blur-2xl shadow-2xl text-gray-200 select-none pointer-events-auto transition-all duration-150"
            >
              <div className="flex items-start justify-between border-b border-white/10 pb-2 mb-2">
                <div>
                  <span className="text-[9px] font-bold tracking-widest text-gray-400 uppercase block">RACK DETAILS</span>
                  <h3 className="text-base font-mono font-bold text-white tracking-wider">{displayRackId}</h3>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`px-2 py-0.5 rounded text-[11px] font-mono font-bold border ${riskBadgeClass}`}>
                    {riskLabel} {riskPct}%
                  </div>
                  <button 
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setHoveredRackId(null);
                      setSelectedRackId(null);
                    }}
                    className="text-gray-400 hover:text-white p-0.5 rounded hover:bg-white/10 transition-colors cursor-pointer text-xs leading-none"
                    title="Close"
                  >
                    ✕
                  </button>
                </div>
              </div>

              <div className="flex items-center gap-1.5 text-[10px] text-gray-400 font-sans mb-2.5 pb-2 border-b border-white/5">
                <span>Zone {zone}</span>
                <span>·</span>
                <span>GNN Node {nodeIndex}</span>
                <span>·</span>
                <span className={isCooling ? 'text-cyan-400 font-semibold' : 'text-gray-400'}>
                  {isCooling ? '❄ COOLING ACTIVE' : 'Cooling Standby'}
                </span>
              </div>

              <div className="p-2 rounded bg-white/[0.04] border border-white/5 mb-2.5">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[9px] font-bold text-gray-400 tracking-wider uppercase">PRIMARY INSIGHT</span>
                  <span className={`text-[10px] font-bold ${recBadgeColor}`}>{recBadge}</span>
                </div>
                <p className="text-[10px] text-gray-300 leading-snug mb-1.5">{recText}</p>
                <div className="flex items-center justify-between text-[10px] font-mono pt-1.5 border-t border-white/5">
                  <span className="text-gray-400 font-sans">Risk Trend: <span className={isRising ? 'text-yellow-400 font-mono' : 'text-green-400 font-mono'}>{riskTrend}</span></span>
                  <span className="text-gray-400 font-sans">Neighbor: <span className="text-cyan-400 font-mono">+{neighborInfluence}%</span></span>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-1.5 font-mono text-[11px] mb-2.5">
                <div className="bg-white/[0.03] p-1.5 rounded border border-white/5 text-center">
                  <div className="text-[9px] text-gray-400 font-sans mb-0.5">CPU</div>
                  <div className="text-cyan-400 font-bold">{cpu.toFixed(1)}%</div>
                </div>
                <div className="bg-white/[0.03] p-1.5 rounded border border-white/5 text-center">
                  <div className="text-[9px] text-gray-400 font-sans mb-0.5">GPU</div>
                  <div className="text-cyan-400 font-bold">{gpu.toFixed(1)}%</div>
                </div>
                <div className="bg-white/[0.03] p-1.5 rounded border border-white/5 text-center">
                  <div className="text-[9px] text-gray-400 font-sans mb-0.5">Temp</div>
                  <div className="text-orange-400 font-bold">{temp.toFixed(1)}°C</div>
                </div>
              </div>

              <div className="space-y-1 font-mono text-[10px] pt-2 border-t border-white/10">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 font-sans">XGBoost Prediction:</span>
                  <span className="text-cyan-300 font-bold">{(xgbPred * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 font-sans">GNN Embedding:</span>
                  <span className="text-purple-300 font-bold">{gnnEmbed.toFixed(4)}</span>
                </div>
                <div className="flex justify-between items-center pt-1 border-t border-white/5">
                  <span className="text-gray-400 font-sans font-medium">Risk Score:</span>
                  <span className={`font-bold ${riskColor}`}>{(riskScore * 100).toFixed(1)}%</span>
                </div>
              </div>

              <div className="mt-3 pt-2.5 border-t border-white/10 flex flex-col gap-2">
                <button
                  type="button"
                  id="hoverBtnDeployCool"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeployCooling(r.id || hoveredRackId);
                  }}
                  className={`w-full py-2 px-3 rounded-lg text-[15.5px] font-semibold transition-all duration-150 cursor-pointer flex items-center justify-center gap-1.5 ${
                    isCooling
                      ? 'bg-cyan-500/25 border border-cyan-400 text-cyan-300 shadow-[0_0_15px_rgba(6,182,212,0.45)] ring-1 ring-cyan-400/50 hover:bg-cyan-500/35 hover:border-cyan-300 hover:text-white hover:shadow-[0_0_20px_rgba(6,182,212,0.6)]'
                      : 'text-white bg-[#1e40af] border border-[#3b82f6]/50 shadow-sm hover:bg-[#2563eb] hover:border-[#60a5fa] hover:shadow-[0_0_14px_rgba(37,99,235,0.45)]'
                  } hover:-translate-y-0.5 active:translate-y-0 active:scale-[0.99]`}
                >
                  <span>Deploy Cooling</span>
                </button>
                <button
                  type="button"
                  id="hoverBtnRemoveCool"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDisengage(r.id || hoveredRackId);
                  }}
                  className={`w-full py-2 px-3 rounded-lg text-[15.5px] font-semibold transition-all duration-150 cursor-pointer flex items-center justify-center gap-1.5 ${
                    !isCooling
                      ? 'bg-white/[0.03] border border-white/10 text-gray-400 hover:bg-white/[0.08] hover:border-white/20 hover:text-gray-300'
                      : 'text-gray-200 bg-white/[0.08] border border-white/25 shadow-sm hover:bg-red-500/20 hover:border-red-400 hover:text-red-200 hover:shadow-[0_0_12px_rgba(239,68,68,0.35)]'
                  } hover:-translate-y-0.5 active:translate-y-0 active:scale-[0.99]`}
                >
                  <span>Disengage</span>
                </button>
              </div>
            </div>
          </Html>
        );
      })()}

      {isReady && (
        <>
          <DataCenterFloor />
          <PerimeterWall />
          <StatsLayer scene={scene} />
          {activeLayer === 'THERMAL' && <ThermalLayer scene={scene} />}
          {activeLayer === 'RISK' && <GNNLayer scene={scene} />}
          {activeLayer === 'AIRFLOW' && <AirflowLayer />}
        </>
      )}
    </group>
  );
}

function PerfExposer() {
  const { gl, scene, camera } = useThree();
  const { setHoveredRackId, setSelectedRackId } = useUiStore();
  React.useEffect(() => {
    (window as any).__gl = gl;
    (window as any).__scene = scene;
    (window as any).__camera = camera;
    (window as any).__THREE = THREE;
    (window as any).__setHoveredRackId = setHoveredRackId;
    (window as any).__setSelectedRackId = setSelectedRackId;
  }, [gl, scene, camera, setHoveredRackId, setSelectedRackId]);
  return null;
}

export function DataCenterScene() {
  return (
    <div className="h-full w-full">
      <Canvas camera={{ position: [22.5, 27.0, 22.5], fov: 42 }} shadows>
        <PerfExposer />
        {/* Deep obsidian data center environment */}
        <color attach="background" args={['#080a0f']} />
        
        {/* Subtle ambient fill */}
        <ambientLight intensity={0.45} color="#e2e8f0" />
        <hemisphereLight args={['#94a3b8', '#080a0f', 0.35]} />

        {/* Primary overhead key light with complete room shadow frustum */}
        <directionalLight
          position={[18, 34, 20]}
          intensity={1.9}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
          shadow-camera-near={0.5}
          shadow-camera-far={70}
          shadow-camera-left={-22}
          shadow-camera-right={22}
          shadow-camera-top={22}
          shadow-camera-bottom={-22}
          shadow-bias={-0.0002}
        />

        {/* Subtle cross-fill light bringing out metallic rack highlights */}
        <directionalLight position={[-16, 22, -14]} intensity={0.60} color="#38bdf8" />
        
        {/* Architectural perimeter downlights casting soft light pools on walls and floor */}
        <pointLight position={[-6.0, 2.9, -11.5]} intensity={0.9} color="#e0f2fe" distance={8} />
        <pointLight position={[6.0, 2.9, -11.5]} intensity={0.9} color="#e0f2fe" distance={8} />
        <pointLight position={[-6.0, 2.9, 11.5]} intensity={0.9} color="#e0f2fe" distance={8} />
        <pointLight position={[6.0, 2.9, 11.5]} intensity={0.9} color="#e0f2fe" distance={8} />
        <pointLight position={[0, 2.8, 12.6]} intensity={1.3} color="#f8fafc" distance={7} />
        <pointLight position={[-8.5, 2.8, 0]} intensity={0.9} color="#e0f2fe" distance={8} />
        <pointLight position={[8.5, 2.8, 0]} intensity={0.9} color="#e0f2fe" distance={8} />

        <Suspense fallback={null}>
          <Model />
          <Environment preset="city" />
        </Suspense>

        <OrbitControls makeDefault target={[0, 1.0, 0]} />
        <CameraController />
      </Canvas>
    </div>
  );
}
