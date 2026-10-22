import { useState, useMemo } from 'react';
import { useTelemetry } from '../services/telemetryApi';
import { Sparkles, Search, Volume2, ExternalLink, CheckCircle, Radio, Loader2 } from 'lucide-react';

interface ExternalContextResult {
  status: 'idle' | 'loading' | 'success' | 'fallback' | 'error';
  is_mock?: boolean;
  message?: string;
  label?: string;
  finding?: string;
  source_title?: string;
  source_domain?: string;
  source_link?: string;
}

interface VoiceBriefResult {
  status: 'idle' | 'loading' | 'ready' | 'fallback' | 'error';
  is_mock?: boolean;
  message?: string;
  label?: string;
  text?: string;
  isPlaying?: boolean;
}

export function AiOperationsCopilot() {
  const { data: telemetry } = useTelemetry();
  
  const [investigateState, setInvestigateState] = useState<ExternalContextResult>({ status: 'idle' });
  const [voiceState, setVoiceState] = useState<VoiceBriefResult>({ status: 'idle' });
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);

  // 1. Identify currently highest-risk rack dynamically from existing application state
  const racks = telemetry?.racks || [];
  const activeRack = useMemo(() => {
    if (racks.length > 0) {
      return [...racks].sort((a: any, b: any) => (b.risk_score || 0) - (a.risk_score || 0))[0];
    }
    // Fallback for single-node / local telemetry mode
    if (telemetry?.risk_score !== undefined || telemetry?.cpu_util !== undefined) {
      return {
        id: 'NODE-01',
        risk_score: telemetry.risk_score ?? 0.45,
        telemetry: {
          cpu_util: telemetry.telemetry?.cpu_util ?? telemetry.cpu_util ?? 48,
          gpu_util: telemetry.telemetry?.gpu_util ?? telemetry.gpu_util ?? 52,
          cpu_temp: telemetry.telemetry?.cpu_temp ?? telemetry.cpu_temp ?? 36.4,
        },
        cooling: {
          status: (telemetry.actual_rpm && telemetry.actual_rpm > 2800) ? 'predictive intervention' : 'normal'
        },
        ai_insights: {
          primary_driver: 'Workload Intensity',
          explanation: 'Chassis thermal risk derived from active CPU and GPU thread intensity.'
        }
      };
    }
    return null;
  }, [telemetry, racks]);

  // Derived rack metrics from authoritative existing data
  const rackId = activeRack?.id || 'RACK-A001';
  const riskScore = activeRack?.risk_score ?? 0.25;
  const riskPct = Math.round(riskScore * 100);

  const isCritical = riskScore >= 0.75;
  const isHigh = riskScore >= 0.55 && !isCritical;
  const isElevated = riskScore >= 0.35 && !isHigh && !isCritical;

  const riskLevel = isCritical ? 'CRITICAL' : isHigh ? 'HIGH' : isElevated ? 'ELEVATED' : 'LOW';
  const riskColor = isCritical ? 'text-red-400' : isHigh ? 'text-orange-400' : isElevated ? 'text-amber-400' : 'text-emerald-400';
  const riskBadgeBorder = isCritical 
    ? 'border-red-500/40 bg-red-500/10 text-red-400' 
    : isHigh 
    ? 'border-orange-500/40 bg-orange-500/10 text-orange-400' 
    : isElevated 
    ? 'border-amber-500/40 bg-amber-500/10 text-amber-400' 
    : 'border-emerald-500/40 bg-emerald-500/10 text-emerald-400';

  const cpuUtil = activeRack?.telemetry?.cpu_util ?? 0;
  const gpuUtil = activeRack?.telemetry?.gpu_util ?? 0;
  const temp = activeRack?.telemetry?.cpu_temp ?? (30 + riskScore * 35);
  const isCoolingActive = activeRack?.cooling?.status === 'predictive intervention' || activeRack?.cooling?.override;
  const coolingStatusLabel = isCoolingActive ? 'PREDICTIVE' : 'STANDBY';

  // 2. AI Explanation using existing data only
  const primaryDriver = activeRack?.ai_insights?.primary_driver || (gpuUtil > cpuUtil ? 'GPU Workload' : 'CPU Thread Execution');
  const aiExplanation = activeRack?.ai_insights?.explanation || 
    `${primaryDriver} is the dominant contributor to predicted thermal risk, with additional influence from neighboring thermal propagation.`;

  // 3. INVESTIGATE Handler (calls SerpAPI backend endpoint)
  const handleInvestigate = async () => {
    setInvestigateState({ status: 'loading' });
    
    const dynamicQuery = `${primaryDriver} high utilization thermal throttling server cooling data center`;
    const payload = {
      rackId,
      riskScore,
      riskLevel,
      cpu: cpuUtil,
      gpu: gpuUtil,
      temperature: temp,
      coolingState: coolingStatusLabel,
      explanation: aiExplanation,
      query: dynamicQuery
    };

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 6000);

      const res = await fetch('http://localhost:8080/api/investigate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setInvestigateState({
        status: data.is_mock ? 'fallback' : 'success',
        is_mock: data.is_mock,
        message: data.message || 'EXTERNAL CONTEXT READY',
        label: data.label,
        finding: data.finding,
        source_title: data.source_title,
        source_domain: data.source_domain,
        source_link: data.source_link
      });
    } catch {
      // Clean isolated development fallback if backend is unreachable
      setInvestigateState({
        status: 'fallback',
        is_mock: true,
        message: 'SerpAPI is not configured.',
        label: 'DEVELOPMENT / MOCK DATA',
        finding: `High ${primaryDriver} load on ${rackId} (${gpuUtil.toFixed(0)}% GPU, ${cpuUtil.toFixed(0)}% CPU) generates localized heat flux. Standard practice requires dynamic CFM acceleration prior to silicon throttle points.`,
        source_title: 'ASHRAE TC 9.9 High-Density Thermal Management Guidelines',
        source_domain: 'tc0909.ashraetcs.org',
        source_link: 'https://tc0909.ashraetcs.org/resources.html'
      });
    }
  };

  // 4. BRIEF ME Handler (calls ElevenLabs backend endpoint)
  const handleBriefMe = async () => {
    // Stop any currently playing audio
    if (audioElement) {
      audioElement.pause();
      setAudioElement(null);
    }
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }

    setVoiceState({ status: 'loading' });

    const briefText = `Thermal alert. Rack ${rackId} currently has a ${riskLevel.toLowerCase()} predicted thermal risk of ${riskPct} percent. ${primaryDriver} is elevated at ${gpuUtil.toFixed(0)} percent GPU and ${cpuUtil.toFixed(0)} percent CPU. ${isCoolingActive ? 'Predictive cooling is engaged' : 'Cooling is currently in standby'}.`;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 7000);

      const res = await fetch('http://localhost:8080/api/voice-brief', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: briefText, rackId, riskScore }),
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (data.audio_base64) {
        // Real ElevenLabs audio returned
        const audio = new Audio(`data:audio/mpeg;base64,${data.audio_base64}`);
        setAudioElement(audio);
        audio.play();
        setVoiceState({
          status: 'ready',
          is_mock: false,
          message: 'Playing ElevenLabs Audio',
          text: briefText,
          isPlaying: true
        });
        audio.onended = () => {
          setVoiceState(prev => ({ ...prev, isPlaying: false }));
        };
      } else {
        // ElevenLabs key not configured - development fallback
        setVoiceState({
          status: 'fallback',
          is_mock: true,
          message: 'ElevenLabs is not configured.',
          label: 'DEVELOPMENT / MOCK DATA',
          text: briefText,
          isPlaying: false
        });
      }
    } catch {
      // Fallback state on network or endpoint issue
      setVoiceState({
        status: 'fallback',
        is_mock: true,
        message: 'ElevenLabs is not configured.',
        label: 'DEVELOPMENT / MOCK DATA',
        text: briefText,
        isPlaying: false
      });
    }
  };

  // Optional local browser speech simulation (only used for dev fallback testing)
  const playBrowserSimulation = (text: string) => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.05;
      utterance.pitch = 1.0;
      utterance.onstart = () => setVoiceState(prev => ({ ...prev, isPlaying: true }));
      utterance.onend = () => setVoiceState(prev => ({ ...prev, isPlaying: false }));
      window.speechSynthesis.speak(utterance);
    }
  };

  return (
    <div className="rounded-xl border border-white/10 bg-[#0B0E14]/90 p-5 backdrop-blur-xl shadow-2xl flex flex-col gap-4 font-sans text-gray-200">
      
      {/* SECTION HEADER */}
      <div className="flex items-center justify-between border-b border-white/10 pb-3">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_#22d3ee] animate-pulse" />
          <h2 className="text-sm font-mono font-bold text-white tracking-wider uppercase">
            AI Operations Copilot
          </h2>
        </div>
        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 font-semibold">
          ACTIVE INTEL
        </span>
      </div>

      {/* A. CURRENT THERMAL RISK */}
      <div className="bg-[#10141D] rounded-lg p-3.5 border border-white/5 flex flex-col gap-2.5">
        <div className="flex justify-between items-center text-[10px] font-mono text-gray-400 uppercase tracking-wider">
          <span>Current Thermal Risk</span>
          <span className="text-gray-300 font-bold">{rackId}</span>
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-baseline gap-2">
            <span className={`text-2xl font-mono font-bold ${riskColor}`}>{riskPct}%</span>
            <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${riskBadgeBorder}`}>
              {riskLevel}
            </span>
          </div>

          <div className="text-right font-mono text-xs">
            <div className="text-[10px] text-gray-400 uppercase">Cooling State</div>
            <div className={`font-bold ${isCoolingActive ? 'text-cyan-400 animate-pulse' : 'text-gray-300'}`}>
              {coolingStatusLabel}
            </div>
          </div>
        </div>

        {/* Telemetry Micro Grid */}
        <div className="grid grid-cols-3 gap-2 pt-2 border-t border-white/5 font-mono text-xs">
          <div className="bg-white/[0.03] p-1.5 rounded border border-white/5 text-center">
            <div className="text-[9px] text-gray-400 uppercase">CPU</div>
            <div className="font-bold text-white mt-0.5">{cpuUtil.toFixed(1)}%</div>
          </div>
          <div className="bg-white/[0.03] p-1.5 rounded border border-white/5 text-center">
            <div className="text-[9px] text-gray-400 uppercase">GPU</div>
            <div className="font-bold text-cyan-300 mt-0.5">{gpuUtil.toFixed(1)}%</div>
          </div>
          <div className="bg-white/[0.03] p-1.5 rounded border border-white/5 text-center">
            <div className="text-[9px] text-gray-400 uppercase">TEMP</div>
            <div className="font-bold text-amber-300 mt-0.5">{temp.toFixed(1)}°C</div>
          </div>
        </div>
      </div>

      {/* B. AI EXPLANATION */}
      <div className="bg-[#10141D] rounded-lg p-3 border border-white/5 flex flex-col gap-1.5">
        <div className="flex items-center gap-1.5 text-[10px] font-mono font-bold text-cyan-400 uppercase tracking-wider">
          <Sparkles size={12} /> AI Risk Explanation
        </div>
        <p className="text-xs font-mono text-gray-300 leading-relaxed text-[11px]">
          "{aiExplanation}"
        </p>
      </div>

      {/* C & D: INTERACTIVE ACTION BUTTONS (INVESTIGATE & BRIEF ME) */}
      <div className="grid grid-cols-2 gap-2.5">
        
        {/* INVESTIGATE BUTTON */}
        <button
          onClick={handleInvestigate}
          disabled={investigateState.status === 'loading'}
          className="flex items-center justify-center gap-2 py-2 px-3 rounded-lg border border-cyan-500/40 bg-cyan-950/40 hover:bg-cyan-900/60 text-cyan-300 font-mono text-xs font-bold transition-all shadow-[0_0_10px_rgba(6,182,212,0.15)] disabled:opacity-50 cursor-pointer"
        >
          {investigateState.status === 'loading' ? (
            <>
              <Loader2 size={13} className="animate-spin text-cyan-400" />
              <span>SEARCHING...</span>
            </>
          ) : (
            <>
              <Search size={13} className="text-cyan-400" />
              <span>INVESTIGATE</span>
            </>
          )}
        </button>

        {/* BRIEF ME BUTTON */}
        <button
          onClick={handleBriefMe}
          disabled={voiceState.status === 'loading'}
          className="flex items-center justify-center gap-2 py-2 px-3 rounded-lg border border-purple-500/40 bg-purple-950/40 hover:bg-purple-900/60 text-purple-300 font-mono text-xs font-bold transition-all shadow-[0_0_10px_rgba(168,85,247,0.15)] disabled:opacity-50 cursor-pointer"
        >
          {voiceState.status === 'loading' ? (
            <>
              <Loader2 size={13} className="animate-spin text-purple-400" />
              <span>SYNTHESIZING...</span>
            </>
          ) : (
            <>
              <Volume2 size={13} className={voiceState.isPlaying ? 'animate-pulse text-purple-300' : 'text-purple-400'} />
              <span>BRIEF ME</span>
            </>
          )}
        </button>
      </div>

      {/* VOICE BRIEF RESULT / STATUS PANEL */}
      {voiceState.status !== 'idle' && (
        <div className="p-3 rounded-lg bg-[#10141D] border border-purple-500/30 flex flex-col gap-2 font-mono text-xs">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5 text-purple-400 font-bold text-[10px] uppercase tracking-wider">
              <Radio size={11} className={voiceState.isPlaying ? 'animate-ping' : ''} />
              Operational Briefing
            </div>
            {voiceState.is_mock && (
              <span className="text-[9px] bg-amber-500/20 text-amber-300 border border-amber-500/30 px-1.5 py-0.2 rounded font-semibold">
                {voiceState.label || 'DEVELOPMENT / MOCK DATA'}
              </span>
            )}
          </div>

          <p className="text-[11px] text-gray-300 leading-relaxed italic bg-white/[0.02] p-2 rounded border border-white/5">
            "{voiceState.text}"
          </p>

          {voiceState.is_mock && (
            <div className="flex items-center justify-between pt-1 border-t border-white/5">
              <span className="text-[10px] text-gray-400">ElevenLabs API key not set</span>
              <button
                onClick={() => voiceState.text && playBrowserSimulation(voiceState.text)}
                className="text-[10px] text-cyan-300 hover:text-cyan-200 underline cursor-pointer"
              >
                {voiceState.isPlaying ? 'Speaking...' : 'Play Audio (Browser Sim)'}
              </button>
            </div>
          )}
        </div>
      )}

      {/* E. EXTERNAL CONTEXT RESULT PANEL */}
      {investigateState.status !== 'idle' && (
        <div className="p-3.5 rounded-lg bg-[#10141D] border border-cyan-500/30 flex flex-col gap-2 font-mono text-xs transition-all">
          
          <div className="flex items-center justify-between">
            <div className="text-[10px] font-bold tracking-wider text-cyan-400 uppercase flex items-center gap-1.5">
              <ExternalLink size={11} /> External Context
            </div>
            
            {investigateState.status === 'loading' ? (
              <span className="text-[10px] text-cyan-300 animate-pulse">INVESTIGATING...</span>
            ) : investigateState.is_mock ? (
              <span className="text-[9px] bg-amber-500/20 text-amber-300 border border-amber-500/30 px-1.5 py-0.2 rounded font-semibold">
                {investigateState.label || 'DEVELOPMENT / MOCK DATA'}
              </span>
            ) : (
              <span className="text-[10px] text-emerald-400 font-bold flex items-center gap-1">
                <CheckCircle size={10} /> CONTEXT READY
              </span>
            )}
          </div>

          {investigateState.status === 'loading' && (
            <div className="text-[11px] text-gray-400 py-2 text-center animate-pulse">
              Querying external engineering reference sources...
            </div>
          )}

          {investigateState.finding && (
            <div className="space-y-2 text-[11px]">
              <p className="text-gray-200 leading-relaxed bg-white/[0.02] p-2.5 rounded border border-white/5">
                {investigateState.finding}
              </p>

              <div className="pt-1.5 border-t border-white/5 flex items-center justify-between text-[10px]">
                <div className="truncate max-w-[200px] text-gray-400">
                  <span className="text-gray-500">Source: </span>
                  <span className="text-white font-semibold">{investigateState.source_title}</span>
                </div>
                {investigateState.source_link && (
                  <a
                    href={investigateState.source_link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-cyan-400 hover:text-cyan-300 underline flex items-center gap-1 shrink-0"
                  >
                    <span>{investigateState.source_domain || 'Link'}</span>
                    <ExternalLink size={9} />
                  </a>
                )}
              </div>
            </div>
          )}
        </div>
      )}

    </div>
  );
}
