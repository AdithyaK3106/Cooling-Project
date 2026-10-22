import React, { useEffect, useState } from 'react';
import { useTelemetry } from '../services/telemetryApi';
import { Activity, Cpu, Thermometer, Fan, Network, Zap, Server, ArrowDown, Laptop } from 'lucide-react';

const FlowArrows = ({ active, color }: { active: boolean, color: string }) => (
    <div className={`flex flex-col items-center justify-center my-4 transition-all duration-500 ${active ? color : 'text-gray-800'}`}>
        <ArrowDown size={28} className={`-mb-3 ${active ? 'animate-bounce' : ''}`} />
        <ArrowDown size={28} className={`-mb-3 opacity-60 ${active ? 'animate-bounce' : ''}`} style={{ animationDelay: '75ms' }} />
        <ArrowDown size={28} className={`opacity-30 ${active ? 'animate-bounce' : ''}`} style={{ animationDelay: '150ms' }} />
    </div>
);

const NeuralNetSVG = ({ active, riskLevel }: { active: boolean, riskLevel: string }) => {
    // A simplified visual representation of a neural network policy
    const color = active ? (riskLevel === 'HIGH' ? '#ef4444' : (riskLevel === 'MEDIUM' ? '#f97316' : '#22c55e')) : '#374151';
    const linkColor = active ? '#6366f1' : '#1f2937';
    
    return (
        <div className="w-full h-48 relative flex justify-center items-center">
            <svg width="400" height="150" viewBox="0 0 400 150">
                {/* Connections Input -> Hidden */}
                {[25, 75, 125].map(y1 => (
                    [30, 60, 90, 120].map(y2 => (
                        <path key={`i-h-${y1}-${y2}`} d={`M 50 ${y1} L 200 ${y2}`} stroke={linkColor} strokeWidth="1.5" opacity="0.4" fill="none" className={active ? 'animate-pulse' : ''} />
                    ))
                ))}
                
                {/* Connections Hidden -> Output */}
                {[30, 60, 90, 120].map(y1 => (
                    <path key={`h-o-${y1}`} d={`M 200 ${y1} L 350 75`} stroke={color} strokeWidth="2" opacity="0.6" fill="none" className={active ? 'animate-pulse' : ''} />
                ))}

                {/* Input Nodes (State Vector) */}
                <circle cx="50" cy="25" r="8" fill="#3b82f6" />
                <circle cx="50" cy="75" r="8" fill="#3b82f6" />
                <circle cx="50" cy="125" r="8" fill="#3b82f6" />
                
                {/* Hidden Nodes */}
                <circle cx="200" cy="30" r="10" fill={active ? '#8b5cf6' : '#374151'} />
                <circle cx="200" cy="60" r="10" fill={active ? '#8b5cf6' : '#374151'} />
                <circle cx="200" cy="90" r="10" fill={active ? '#8b5cf6' : '#374151'} />
                <circle cx="200" cy="120" r="10" fill={active ? '#8b5cf6' : '#374151'} />

                {/* Output Node (Risk Level) */}
                <circle cx="350" cy="75" r="14" fill={color} className={active ? 'animate-pulse' : ''} />
                <text x="350" y="79" textAnchor="middle" fill="#000" fontSize="10" fontWeight="bold">ACT</text>
            </svg>
        </div>
    );
};

export function LocalPipelineTab() {
    const { data: telemetry } = useTelemetry();
    
    const isLive = telemetry?.live_mode || false;
    const connected2 = telemetry?.node2_connected ?? false;
    
    const risk1 = telemetry?.risk_score || 0.0;
    const fan1 = telemetry?.fan_percent || 0.0;
    const mode1 = telemetry?.risk_level || 'BASELINE';
    
    const risk2 = telemetry?.node2_propagated_risk || 0.0;
    const fan2 = telemetry?.node2_fan || 0.0;

    const getColor = (val: number | undefined, threshold: number) => (val || 0) > threshold ? 'text-red-400' : 'text-cyan-400';

    return (
        <div className="flex flex-col h-full bg-[#05070a] p-6 overflow-y-auto">
            
            <div className="mb-6 flex justify-between items-center shrink-0">
                <div>
                    <h1 className="text-3xl font-bold text-white tracking-widest flex items-center gap-3">
                        <Network className="text-purple-500" size={32} /> ML PIPELINE VISUALIZATION
                    </h1>
                    <p className="text-gray-400 mt-2">Top-down flowchart of the live THERVO inference engine</p>
                </div>
            </div>

            {/* FLOWCHART CONTAINER */}
            <div className="max-w-4xl mx-auto w-full pb-20 pt-4">
                
                {/* 1. HARDWARE TELEMETRY STAGE */}
                <div className="bg-black/40 border border-white/10 rounded-2xl p-8 relative shadow-[0_0_30px_rgba(0,0,0,0.5)]">
                    <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-blue-500/20 border border-blue-500 text-blue-400 px-4 py-1 rounded-full text-xs font-bold tracking-widest flex items-center gap-2">
                        <Server size={14} /> STAGE 1: HARDWARE TELEMETRY
                    </div>
                    
                    <div className="grid grid-cols-2 gap-8 mt-4">
                        {/* Node 1 */}
                        <div className={`border ${risk1 > 0.6 ? 'border-red-500/50 shadow-[0_0_20px_rgba(239,68,68,0.2)]' : 'border-blue-500/30'} bg-white/5 rounded-xl p-6 transition-all duration-500`}>
                            <div className="flex items-center gap-3 mb-4 border-b border-white/10 pb-3">
                                <Laptop className="text-blue-400" size={24} />
                                <div>
                                    <h3 className="font-bold text-white tracking-widest">NODE 1 (MAIN)</h3>
                                    <div className="text-[10px] text-gray-400">Lenovo LOQ / Localhost</div>
                                </div>
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-black/50 rounded p-3 text-center border border-white/5">
                                    <Cpu className="mx-auto mb-2 text-gray-500" size={18} />
                                    <div className={`text-xl font-bold ${getColor(telemetry?.cpu, 80)}`}>{(telemetry?.cpu || 0).toFixed(0)}%</div>
                                    <div className="text-[10px] text-gray-500 uppercase tracking-widest">CPU LOAD</div>
                                </div>
                                <div className="bg-black/50 rounded p-3 text-center border border-white/5">
                                    <Thermometer className="mx-auto mb-2 text-gray-500" size={18} />
                                    <div className={`text-xl font-bold ${getColor(telemetry?.cpu_temp, 85)}`}>{(telemetry?.cpu_temp || 0).toFixed(0)}°C</div>
                                    <div className="text-[10px] text-gray-500 uppercase tracking-widest">CPU TEMP</div>
                                </div>
                            </div>
                        </div>

                        {/* Node 2 */}
                        <div className={`border ${connected2 ? 'border-green-500/30' : 'border-gray-800'} bg-white/5 rounded-xl p-6 transition-all duration-500 ${!connected2 ? 'opacity-50 grayscale' : ''}`}>
                            <div className="flex items-center gap-3 mb-4 border-b border-white/10 pb-3">
                                <Laptop className={connected2 ? "text-green-400" : "text-gray-500"} size={24} />
                                <div>
                                    <h3 className="font-bold text-white tracking-widest">NODE 2 (NEIGHBOR)</h3>
                                    <div className="text-[10px] text-gray-400">{connected2 ? 'Acer / Connected' : 'Disconnected'}</div>
                                </div>
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="bg-black/50 rounded p-3 text-center border border-white/5">
                                    <Cpu className="mx-auto mb-2 text-gray-500" size={18} />
                                    <div className="text-xl font-bold text-white">{(telemetry?.node2_cpu || 0).toFixed(0)}%</div>
                                    <div className="text-[10px] text-gray-500 uppercase tracking-widest">CPU LOAD</div>
                                </div>
                                <div className="bg-black/50 rounded p-3 text-center border border-white/5">
                                    <Thermometer className="mx-auto mb-2 text-gray-500" size={18} />
                                    <div className="text-xl font-bold text-white">{(telemetry?.node2_cpu_temp || 0).toFixed(0)}°C</div>
                                    <div className="text-[10px] text-gray-500 uppercase tracking-widest">CPU TEMP</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <FlowArrows active={true} color="text-blue-500" />

                {/* 2. GNN STAGE */}
                <div className="bg-black/40 border border-white/10 rounded-2xl p-8 relative shadow-[0_0_30px_rgba(0,0,0,0.5)]">
                    <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-purple-500/20 border border-purple-500 text-purple-400 px-4 py-1 rounded-full text-xs font-bold tracking-widest flex items-center gap-2">
                        <Network size={14} /> STAGE 2: GRAPH NEURAL NETWORK
                    </div>
                    
                    <div className="mt-4 p-6 bg-white/5 border border-white/10 rounded-xl relative">
                        {/* Status Label */}
                        <div className="absolute top-4 left-4 text-xs font-mono">
                            <span className="text-gray-500">GNN Status: </span>
                            {isLive ? <span className="text-purple-400 font-bold animate-pulse">PROPAGATING</span> : <span className="text-gray-500">BYPASSED (LOCAL MODE)</span>}
                        </div>

                        <div className="flex items-center justify-between mt-8 px-12">
                            {/* N1 */}
                            <div className="flex flex-col items-center">
                                <div className={`w-20 h-20 rounded-full border-4 ${risk1 > 0.6 ? 'border-red-500 bg-red-500/20' : 'border-blue-500 bg-blue-500/20'} flex items-center justify-center shadow-[0_0_20px_rgba(59,130,246,0.3)] transition-all duration-500`}>
                                    <span className="font-bold text-xl">N1</span>
                                </div>
                                <div className="mt-4 text-center">
                                    <div className="text-xs text-gray-400">Risk Embedding</div>
                                    <div className="font-mono text-white">[{(risk1).toFixed(2)}]</div>
                                </div>
                            </div>

                            {/* Edge */}
                            <div className="flex-1 flex flex-col items-center justify-center relative -mt-8">
                                <div className="text-xs font-mono text-purple-400 mb-2 font-bold">{isLive ? 'EDGE WEIGHT: 0.50' : 'ISOLATED'}</div>
                                <div className="w-full h-1 bg-gray-800 rounded-full relative overflow-hidden mx-4">
                                    {isLive && connected2 && (
                                        <div className="absolute top-0 left-0 h-full bg-purple-500 w-full animate-pulse shadow-[0_0_10px_#a855f7]"></div>
                                    )}
                                </div>
                            </div>

                            {/* N2 */}
                            <div className="flex flex-col items-center">
                                <div className={`w-20 h-20 rounded-full border-4 ${!connected2 ? 'border-gray-700 bg-gray-800' : (risk2 > 0.6 ? 'border-red-500 bg-red-500/20' : 'border-green-500 bg-green-500/20')} flex items-center justify-center transition-all duration-500`}>
                                    <span className="font-bold text-xl">N2</span>
                                </div>
                                <div className="mt-4 text-center">
                                    <div className="text-xs text-gray-400">Propagated Risk</div>
                                    <div className="font-mono text-white">{(risk2*100).toFixed(1)}%</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <FlowArrows active={isLive} color="text-purple-500" />

                {/* 3. POLICY INFERENCE STAGE */}
                <div className="bg-black/40 border border-white/10 rounded-2xl p-8 relative shadow-[0_0_30px_rgba(0,0,0,0.5)]">
                    <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-orange-500/20 border border-orange-500 text-orange-400 px-4 py-1 rounded-full text-xs font-bold tracking-widest flex items-center gap-2">
                        <Zap size={14} /> STAGE 3: POLICY INFERENCE
                    </div>
                    
                    <div className="mt-4 bg-white/5 border border-white/10 rounded-xl flex items-center overflow-hidden">
                        <div className="w-1/3 p-6 border-r border-white/10 bg-black/40">
                            <h4 className="text-[10px] font-bold tracking-widest text-gray-500 mb-4">COMBINED STATE VECTOR</h4>
                            <div className="space-y-3 font-mono text-xs">
                                <div className="flex justify-between"><span className="text-gray-400">Local Risk</span><span className="text-blue-400">{(risk1).toFixed(3)}</span></div>
                                <div className="flex justify-between"><span className="text-gray-400">Neighbor Risk</span><span className="text-purple-400">{(risk2).toFixed(3)}</span></div>
                                <div className="flex justify-between"><span className="text-gray-400">CPU/GPU Delta</span><span className="text-cyan-400">+12°C</span></div>
                            </div>
                        </div>
                        
                        <div className="w-1/3 border-r border-white/10 bg-black/60">
                            <NeuralNetSVG active={true} riskLevel={mode1} />
                        </div>

                        <div className="w-1/3 p-6 bg-black/40 text-center flex flex-col items-center justify-center h-full">
                            <h4 className="text-[10px] font-bold tracking-widest text-gray-500 mb-4">INFERENCE DECISION</h4>
                            <div className={`inline-block border-2 ${mode1 === 'HIGH' ? 'border-red-500 text-red-500 bg-red-500/10 shadow-[0_0_20px_rgba(239,68,68,0.4)]' : (mode1 === 'MEDIUM' ? 'border-orange-500 text-orange-500 bg-orange-500/10 shadow-[0_0_20px_rgba(249,115,22,0.4)]' : 'border-green-500 text-green-500 bg-green-500/10 shadow-[0_0_20px_rgba(34,197,94,0.4)]')} rounded-xl px-6 py-4 text-2xl font-black tracking-widest transition-all duration-500`}>
                                {mode1}
                            </div>
                            <div className="text-[10px] text-gray-400 mt-4">Mapped to Actuator Space</div>
                        </div>
                    </div>
                </div>

                <FlowArrows active={true} color={mode1 === 'HIGH' ? 'text-red-500' : (mode1 === 'MEDIUM' ? 'text-orange-500' : 'text-green-500')} />

                {/* 4. HARDWARE ACTUATION STAGE */}
                <div className="bg-black/40 border border-white/10 rounded-2xl p-8 relative shadow-[0_0_30px_rgba(0,0,0,0.5)]">
                    <div className="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-green-500/20 border border-green-500 text-green-400 px-4 py-1 rounded-full text-xs font-bold tracking-widest flex items-center gap-2">
                        <Fan size={14} /> STAGE 4: HARDWARE ACTUATION
                    </div>

                    <div className="grid grid-cols-2 gap-8 mt-4">
                        {/* Node 1 Actuator */}
                        <div className="border border-green-500/30 bg-green-500/5 rounded-xl p-6 relative overflow-hidden transition-all duration-500">
                            <div className="absolute top-4 right-4 text-[10px] font-bold tracking-widest text-green-500 border border-green-500/50 bg-green-500/10 px-2 py-1 rounded">WMI / LLT ENFORCED</div>
                            <div className="flex items-center gap-4 mb-4 mt-2">
                                <div className="bg-green-500/20 p-3 rounded-lg border border-green-500/30">
                                    <Fan className={`text-green-400 ${fan1 > 50 ? 'animate-spin' : ''}`} size={24} style={{ animationDuration: fan1 > 80 ? '0.5s' : '2s' }} />
                                </div>
                                <div>
                                    <h3 className="font-bold text-white tracking-widest">NODE 1 FANS</h3>
                                    <div className="text-xs font-mono text-cyan-400">Target: {fan1?.toFixed(1)}%</div>
                                </div>
                            </div>
                            <div className="w-full bg-black/50 h-2 rounded-full overflow-hidden border border-white/5">
                                <div className="h-full bg-gradient-to-r from-green-500 to-cyan-500 transition-all duration-1000" style={{ width: `${fan1}%` }}></div>
                            </div>
                        </div>

                        {/* Node 2 Actuator */}
                        <div className={`border ${connected2 ? 'border-green-500/30 bg-green-500/5' : 'border-gray-800 bg-white/5'} rounded-xl p-6 relative overflow-hidden transition-all duration-500 ${!connected2 ? 'opacity-50 grayscale' : ''}`}>
                            {connected2 && <div className="absolute top-4 right-4 text-[10px] font-bold tracking-widest text-green-500 border border-green-500/50 bg-green-500/10 px-2 py-1 rounded">REMOTE ENFORCED</div>}
                            <div className="flex items-center gap-4 mb-4 mt-2">
                                <div className={`${connected2 ? 'bg-green-500/20 border-green-500/30 text-green-400' : 'bg-gray-800 border-gray-700 text-gray-500'} p-3 rounded-lg border`}>
                                    <Fan className={`${connected2 && fan2 > 50 ? 'animate-spin' : ''}`} size={24} style={{ animationDuration: fan2 > 80 ? '0.5s' : '2s' }} />
                                </div>
                                <div>
                                    <h3 className="font-bold text-white tracking-widest">NODE 2 FANS</h3>
                                    <div className="text-xs font-mono text-cyan-400">Target: {fan2?.toFixed(1)}%</div>
                                </div>
                            </div>
                            <div className="w-full bg-black/50 h-2 rounded-full overflow-hidden border border-white/5">
                                <div className="h-full bg-gradient-to-r from-green-500 to-cyan-500 transition-all duration-1000" style={{ width: `${fan2}%` }}></div>
                            </div>
                        </div>
                    </div>
                </div>

            </div>
        </div>
    );
}
