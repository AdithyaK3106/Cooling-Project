import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
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
import * as THREE from 'three';

function Model() {
  const { scene } = useGLTF('/models/room_server.glb');
  const { setSelectedRackId, setHoveredRackId, activeLayer } = useUiStore();
  const hoveredRackId = useUiStore((state) => state.hoveredRackId);
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
  }, [hoveredRackId]);

  const [isReady, setIsReady] = React.useState(false);

  // Traverse the scene once to assign IDs, attach event handlers, and STRIP excess racks
  React.useEffect(() => {
    const nodesToRemove: THREE.Object3D[] = [];

    scene.traverse((child: any) => {
      // Hide roof/ceiling so we can see inside and click
      if (child.name.toLowerCase().includes('roof') || child.name.toLowerCase().includes('ceiling') || child.name.toLowerCase().includes('top plane')) {
        child.visible = false;
      }

      // Check if this object or any ancestor is a Rack
      let isPartOfRack = false;
      let node = child;
      while (node) {
        if (node.name.match(/Rack/i)) {
          isPartOfRack = true;
          break;
        }
        node = node.parent;
      }

      // Disable raycasting on non-rack objects (walls, floors) so hover works!
      if (!isPartOfRack) {
        child.raycast = () => null;
      }

      const match = child.name.match(/^Rack\s*(\d+)$/i) || child.name.match(/^Rack_(\d+)$/i);
      if (match) {
        const num = parseInt(match[1], 10);
        
        // Strip out everything except the first 25 racks
        if (num > 25) {
          nodesToRemove.push(child);
          return;
        }

        const rackId = `A0${num}`;
        child.userData = { rackId };

        // Force into a perfect 5x5 grid centered at origin
        const index = num - 1;
        const row = Math.floor(index / 5);
        const col = index % 5;
        
        // Adjust spacing based on visual preference
        const spacingX = 3.5;
        const spacingZ = 5.0;
        
        child.position.set(
          (col - 2) * spacingX,
          0,
          (row - 2) * spacingZ
        );
        // Reset rotation so they all face forward neatly
        child.rotation.set(0, 0, 0);

        // Hide original low-poly placeholder meshes
        child.children.forEach((c: any) => {
          if (!c.userData?.isDetailedCabinet) {
            c.visible = false;
          }
        });

        // Attach realistic enterprise server cabinet
        if (!child.children.some((c: any) => c.userData?.isDetailedCabinet)) {
          const cabinet = createDetailedServerRack(rackId);
          child.add(cabinet);
        }
      }
    });

    // Remove the excess racks from the scene graph entirely
    nodesToRemove.forEach(node => node.removeFromParent());
    
    setIsReady(true);
  }, [scene]);

  // Find the position of the hovered rack for the tooltip
  const hoveredRackPos = React.useMemo(() => {
    if (!hoveredRackId) return null;
    let pos = new THREE.Vector3();
    scene.traverse((child) => {
      if (child.userData.rackId === hoveredRackId) {
        child.getWorldPosition(pos);
        pos.y += 1.2; // Level with rack midpoint for clear lateral projection
      }
    });
    return pos;
  }, [hoveredRackId, scene]);

  // Ensure the simulated datacenter fleet is actively generating dynamic telemetry
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
        onClick={(e: any) => {
          e.stopPropagation();
          let node = e.object;
          while (node && !(node.name.match(/^Rack\s*\d+$/i) || node.name.match(/^Rack_\d+$/i)) && node.parent) {
            node = node.parent;
          }
          if (node && (node.name.match(/^Rack\s*\d+$/i) || node.name.match(/^Rack_\d+$/i))) {
            setSelectedRackId(node.userData.rackId);
          } else {
            setSelectedRackId(null);
          }
        }}
        onPointerOver={(e: any) => {
          e.stopPropagation();
          let node = e.object;
          while (node && !(node.name.match(/^Rack\s*\d+$/i) || node.name.match(/^Rack_\d+$/i)) && node.parent) {
            node = node.parent;
          }
          if (node && (node.name.match(/^Rack\s*\d+$/i) || node.name.match(/^Rack_\d+$/i))) {
            clearHoverTimeout();
            isOverCardRef.current = false;
            setHoveredRackId(node.userData.rackId);
            document.body.style.cursor = 'pointer';
          }
        }}
        onPointerOut={(e: any) => {
          e.stopPropagation();
          clearHoverTimeout();
          hoverTimeoutRef.current = setTimeout(() => {
            if (!isOverCardRef.current) {
              setHoveredRackId(null);
              document.body.style.cursor = 'default';
            }
          }, 300);
        }}
        onPointerMissed={() => {
          setSelectedRackId(null);
          if (!isOverCardRef.current) {
            setHoveredRackId(null);
          }
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
                clearHoverTimeout();
                hoverTimeoutRef.current = setTimeout(() => {
                  if (!isOverCardRef.current) {
                    setHoveredRackId(null);
                    document.body.style.cursor = 'default';
                  }
                }, 200);
              }}
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => e.stopPropagation()}
              className="w-72 rounded-xl border border-white/15 bg-[#0B0E14]/95 p-3.5 backdrop-blur-2xl shadow-2xl text-gray-200 select-none pointer-events-auto transition-all duration-150"
            >
              {/* Header: Tag & ID & Risk Badge */}
              <div className="flex items-start justify-between border-b border-white/10 pb-2 mb-2">
                <div>
                  <span className="text-[9px] font-bold tracking-widest text-gray-400 uppercase block">HOVERED RACK</span>
                  <h3 className="text-base font-mono font-bold text-white tracking-wider">{displayRackId}</h3>
                </div>
                <div className={`px-2 py-0.5 rounded text-[11px] font-mono font-bold border ${riskBadgeClass}`}>
                  {riskLabel} {riskPct}%
                </div>
              </div>

              {/* Sub-line: Zone · GNN Node · Cooling status */}
              <div className="flex items-center gap-1.5 text-[10px] text-gray-400 font-sans mb-2.5 pb-2 border-b border-white/5">
                <span>Zone {zone}</span>
                <span>·</span>
                <span>GNN Node {nodeIndex}</span>
                <span>·</span>
                <span className={isCooling ? 'text-cyan-400 font-semibold' : 'text-gray-400'}>
                  {isCooling ? '❄ COOLING ACTIVE' : 'Cooling Standby'}
                </span>
              </div>

              {/* Primary Insight Box */}
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

              {/* Live Telemetry (CPU, GPU, Temperature) */}
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

              {/* AI Predictions & Risk Score */}
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

              {/* Action Buttons: Deploy Cooling & Disengage */}
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

import { useThree } from '@react-three/fiber';

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
      <Canvas camera={{ position: [50, 50, 50], fov: 45 }}>
        <PerfExposer />
        <color attach="background" args={['#1A1C23']} />
        
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 20, 10]} intensity={1.5} />
        
        <Suspense fallback={null}>
          <Model />
          <Environment preset="city" />
        </Suspense>

        <OrbitControls makeDefault />
        <CameraController />
      </Canvas>
    </div>
  );
}
