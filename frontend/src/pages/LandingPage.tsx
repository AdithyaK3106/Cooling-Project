import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Box } from 'lucide-react';

export interface LandingPageProps {
  /** Video file path relative to public directory (default: '/use_this.mp4') */
  videoSrc?: string;
  /** Route path to navigate to when Enter is clicked (default: '/dual-node') */
  enterTargetRoute?: string;
  /** Custom logo title text (default: 'THERVO') */
  brandName?: string;
}

export const LandingPage: React.FC<LandingPageProps> = ({
  videoSrc = '/use_this.mp4',
  enterTargetRoute = '/dual-node',
  brandName = 'THERVO',
}) => {
  const navigate = useNavigate();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [isVideoLoaded, setIsVideoLoaded] = useState<boolean>(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const handleEnter = () => {
    navigate(enterTargetRoute);
  };

  // Ensure video reliably autoplays
  useEffect(() => {
    const video = videoRef.current;
    if (video) {
      video.muted = true;
      video.defaultMuted = true;
      const playPromise = video.play();
      if (playPromise !== undefined) {
        playPromise
          .then(() => setIsVideoLoaded(true))
          .catch(() => {
            // Autoplay policy or loading
          });
      }
    }
  }, [videoSrc]);

  // Cybernetic particle network canvas fallback when video is buffering or unavailable
  useEffect(() => {
    if (isVideoLoaded) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', handleResize);

    const numNodes = Math.floor((width * height) / 18000);
    const nodes: Array<{ x: number; y: number; vx: number; vy: number; radius: number; color: string }> = [];
    const colors = ['#06b6d4', '#0ea5e9', '#38bdf8', '#0284c7'];

    for (let i = 0; i < numNodes; i++) {
      nodes.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.45,
        vy: (Math.random() - 0.5) * 0.45,
        radius: Math.random() * 2 + 1,
        color: colors[Math.floor(Math.random() * colors.length)],
      });
    }

    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      const grad = ctx.createRadialGradient(width / 2, height / 2, 50, width / 2, height / 2, Math.max(width, height) * 0.7);
      grad.addColorStop(0, '#0a101d');
      grad.addColorStop(0.5, '#060a12');
      grad.addColorStop(1, '#020408');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, width, height);

      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 130) {
            ctx.strokeStyle = `rgba(6, 182, 212, ${0.22 * (1 - dist / 130)})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.stroke();
          }
        }
      }

      nodes.forEach((n) => {
        n.x += n.vx;
        n.y += n.vy;

        if (n.x < 0 || n.x > width) n.vx *= -1;
        if (n.y < 0 || n.y > height) n.vy *= -1;

        ctx.fillStyle = n.color;
        ctx.shadowColor = n.color;
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      });

      animId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
    };
  }, [isVideoLoaded]);

  return (
    <div className="relative w-full h-screen overflow-hidden bg-black text-white font-sans select-none">
      {/* 1. BACKGROUND VIDEO LAYER (Enhanced Brightness & Vibrancy) */}
      <video
        ref={videoRef}
        autoPlay
        loop
        muted
        playsInline
        onLoadedData={() => setIsVideoLoaded(true)}
        onCanPlay={() => setIsVideoLoaded(true)}
        onPlaying={() => setIsVideoLoaded(true)}
        onError={() => setIsVideoLoaded(false)}
        style={{ filter: 'brightness(1.24) contrast(1.05)' }}
        className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-1000 ${
          isVideoLoaded ? 'opacity-100' : 'opacity-0'
        }`}
      >
        <source src={videoSrc} type="video/mp4" />
      </video>

      {/* Very light ambient vignette overlay so the video stays bright and vivid */}
      <div className={`absolute inset-0 bg-gradient-to-b from-black/25 via-transparent to-black/35 pointer-events-none transition-opacity duration-1000 ${
        isVideoLoaded ? 'opacity-100' : 'opacity-0'
      }`} />

      {/* 2. FALLBACK CANVAS BACKGROUND WHEN VIDEO IS LOADING */}
      {!isVideoLoaded && (
        <div className="absolute inset-0">
          <canvas ref={canvasRef} className="absolute inset-0 w-full h-full block" />
          <div 
            className="absolute inset-0 opacity-15 pointer-events-none" 
            style={{
              backgroundImage: `radial-gradient(circle at 1px 1px, rgba(56, 189, 248, 0.5) 1px, transparent 0)`,
              backgroundSize: '40px 40px'
            }}
          />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[700px] bg-cyan-500/10 rounded-full blur-3xl pointer-events-none animate-pulse" />
        </div>
      )}

      {/* 3. BRAND LOGO (TOP CENTER) - No background container */}
      <div className="absolute top-10 left-1/2 -translate-x-1/2 z-20">
        <div 
          className="flex items-center gap-3.5 cursor-pointer group select-none transition-transform duration-300 hover:scale-105"
          onClick={handleEnter}
        >
          <Box className="w-8 h-8 md:w-10 md:h-10 text-cyan-400 drop-shadow-[0_0_16px_rgba(6,182,212,0.8)] group-hover:text-cyan-300 transition-colors duration-300 stroke-[2.2]" />
          <span className="font-black tracking-[0.32em] text-white text-3xl md:text-5xl uppercase drop-shadow-[0_2px_12px_rgba(0,0,0,0.9)] drop-shadow-[0_0_24px_rgba(6,182,212,0.7)] group-hover:text-cyan-200 transition-colors duration-300">
            {brandName}
          </span>
        </div>
      </div>

      {/* 4. ENTER ACTION BUTTON AT BOTTOM CENTER - No background container */}
      <div className="absolute bottom-12 left-1/2 -translate-x-1/2 z-30">
        <button
          onClick={handleEnter}
          className="group flex items-center gap-4 px-6 py-2.5 bg-transparent border-0 text-cyan-300 hover:text-white font-mono font-bold text-xl md:text-2xl tracking-[0.3em] uppercase transition-all duration-300 hover:scale-105 active:scale-95 cursor-pointer drop-shadow-[0_2px_10px_rgba(0,0,0,0.9)] drop-shadow-[0_0_20px_rgba(6,182,212,0.6)]"
        >
          <span className="relative">
            ENTER
            <span className="absolute -bottom-1.5 left-0 w-0 h-0.5 bg-cyan-400 group-hover:w-full transition-all duration-300 shadow-[0_0_8px_rgba(6,182,212,0.9)]" />
          </span>
          <ArrowRight className="w-6 h-6 md:w-7 md:h-7 text-cyan-400 group-hover:text-cyan-200 group-hover:translate-x-2 transition-all duration-300 drop-shadow-[0_0_12px_rgba(6,182,212,0.8)]" />
        </button>
      </div>
    </div>
  );
};

export default LandingPage;
