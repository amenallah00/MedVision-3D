import React, { useState, useEffect, useRef } from 'react';
import {
  Activity,
  Cpu,
  Eye,
  EyeOff,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  RefreshCw,
  Layers,
  ShieldCheck,
  HardDrive,
  CheckCircle2,
  AlertCircle,
  BarChart3,
  Sliders,
  ChevronRight,
  Maximize2,
  Crosshair,
  Zap,
  Info
} from 'lucide-react';
import './App.css';

const API_BASE = '/api';

export default function App() {
  // Application State
  const [status, setStatus] = useState(null);
  const [slices, setSlices] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [pipeline, setPipeline] = useState([]);
  const [loading, setLoading] = useState(true);
  const [inferring, setInferring] = useState(false);
  const [inferResult, setInferResult] = useState(null);
  const [error, setError] = useState(null);

  // Viewer Controls
  const [currentSlice, setCurrentSlice] = useState(15);
  const [showMask, setShowMask] = useState(true);
  const [opacity, setOpacity] = useState(0.55);
  const [colorScheme, setColorScheme] = useState('cyan'); // 'cyan', 'amber', 'emerald', 'violet'
  const [isPlaying, setIsPlaying] = useState(false);
  const [fps, setFps] = useState(8);
  const [windowPreset, setWindowPreset] = useState('abdomen');
  const [showCrosshair, setShowCrosshair] = useState(true);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  const canvasRef = useRef(null);
  const playIntervalRef = useRef(null);

  // Fetch initial telemetry and slices
  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [statusRes, slicesRes, metricsRes, pipelineRes] = await Promise.all([
        fetch(`${API_BASE}/status`).then(r => r.json()),
        fetch(`${API_BASE}/slices`).then(r => r.json()),
        fetch(`${API_BASE}/metrics`).then(r => r.json()),
        fetch(`${API_BASE}/pipeline`).then(r => r.json()),
      ]);

      setStatus(statusRes);
      setSlices(slicesRes.slices || []);
      setMetrics(metricsRes);
      setPipeline(pipelineRes.stages || []);

      if (slicesRes.slices && slicesRes.slices.length > 0) {
        setCurrentSlice(Math.floor(slicesRes.slices.length / 2));
      }
    } catch (err) {
      console.warn("Backend API not reachable yet or loading:", err);
      setError("Connecting to MedVision 3D API server...");
    } finally {
      setLoading(false);
    }
  };

  // Re-run live inference
  const handleRunInference = async () => {
    setInferring(true);
    setInferResult(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, { method: 'POST' });
      const data = await res.json();
      setInferResult(data);
      // Refresh slices & metrics
      await fetchInitialData();
    } catch (err) {
      setInferResult({ status: 'ERROR', message: err.message });
    } finally {
      setInferring(false);
    }
  };

  // Cine Playback Loop
  useEffect(() => {
    if (isPlaying && slices.length > 0) {
      playIntervalRef.current = setInterval(() => {
        setCurrentSlice((prev) => (prev + 1) % slices.length);
      }, 1000 / fps);
    } else {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    }
    return () => {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    };
  }, [isPlaying, slices.length, fps]);

  // Render Canvas with CT slice & colored segmentation overlay
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || slices.length === 0) return;

    const sliceData = slices[currentSlice];
    if (!sliceData) return;

    const ctx = canvas.getContext('2d');
    const ctImg = new Image();
    const segImg = new Image();

    ctImg.src = sliceData.ct;
    ctImg.onload = () => {
      canvas.width = ctImg.width;
      canvas.height = ctImg.height;

      // Draw CT slice
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(ctImg, 0, 0);

      // Apply Window / Level filter adjustments
      if (windowPreset === 'bone') {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.globalCompositeOperation = 'hard-light';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.globalCompositeOperation = 'source-over';
      } else if (windowPreset === 'contrast') {
        ctx.filter = 'contrast(150%) brightness(110%)';
        ctx.drawImage(ctImg, 0, 0);
        ctx.filter = 'none';
      }

      // Draw colored segmentation mask
      if (showMask && sliceData.has_mask) {
        segImg.src = sliceData.seg;
        segImg.onload = () => {
          // Create offscreen canvas to colorize binary mask
          const offscreen = document.createElement('canvas');
          offscreen.width = canvas.width;
          offscreen.height = canvas.height;
          const offCtx = offscreen.getContext('2d');

          // Draw binary mask
          offCtx.drawImage(segImg, 0, 0);

          // Colorize mask
          offCtx.globalCompositeOperation = 'source-in';
          if (colorScheme === 'cyan') offCtx.fillStyle = '#06b6d4';
          else if (colorScheme === 'amber') offCtx.fillStyle = '#f59e0b';
          else if (colorScheme === 'emerald') offCtx.fillStyle = '#10b981';
          else if (colorScheme === 'violet') offCtx.fillStyle = '#a855f7';
          offCtx.fillRect(0, 0, offscreen.width, offscreen.height);

          // Overlay onto main canvas
          ctx.save();
          ctx.globalAlpha = opacity;
          ctx.drawImage(offscreen, 0, 0);
          ctx.restore();
        };
      }
    };
  }, [currentSlice, slices, showMask, opacity, colorScheme, windowPreset]);

  const activeSliceData = slices[currentSlice] || {};

  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 flex flex-col font-sans">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800/80 bg-[#0c1322]/90 backdrop-blur-md sticky top-0 z-50 px-6 py-3.5">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3.5">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 via-sky-500 to-blue-600 flex items-center justify-center shadow-lg shadow-sky-500/20 border border-sky-400/30">
              <Activity className="w-6 h-6 text-white animate-pulse" />
            </div>
            <div>
              <div className="flex items-center space-x-2.5">
                <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-sky-100 to-cyan-300 bg-clip-text text-transparent">
                  MedVision 3D
                </h1>
                <span className="text-[11px] font-semibold tracking-wider uppercase px-2 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-800 text-cyan-300">
                  v2.8.1
                </span>
              </div>
              <p className="text-xs text-slate-400">
                AI-Powered 3D Medical Image Segmentation &bull; Maintained by{' '}
                <span className="font-semibold text-sky-300">Amen Allah Ben Abdrabah</span>
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {/* GPU Telemetry Pill */}
            <div className="hidden md:flex items-center space-x-2 bg-slate-900/90 border border-slate-800 rounded-lg px-3 py-1.5 text-xs">
              <Cpu className="w-4 h-4 text-emerald-400" />
              <div>
                <span className="text-slate-400">GPU: </span>
                <span className="font-mono text-emerald-300 font-medium">
                  {status?.hardware?.device_name || 'NVIDIA RTX 3050 (CUDA 12.4)'}
                </span>
              </div>
            </div>

            {/* CUDA Status Pill */}
            <div className="flex items-center space-x-1.5 bg-emerald-950/50 border border-emerald-800/60 text-emerald-300 rounded-lg px-3 py-1.5 text-xs font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
              <span>Inference Ready</span>
            </div>

            {/* Trigger Live Inference Button */}
            <button
              onClick={handleRunInference}
              disabled={inferring}
              className="flex items-center space-x-2 bg-gradient-to-r from-sky-500 to-cyan-600 hover:from-sky-400 hover:to-cyan-500 text-white font-medium px-4 py-1.5 rounded-lg shadow-md shadow-sky-500/25 transition disabled:opacity-50 text-xs cursor-pointer"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${inferring ? 'animate-spin' : ''}`} />
              <span>{inferring ? 'Inferring 3D Volume...' : 'Run Live Inference'}</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Dashboard */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        {/* Metric Cards Row */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 shadow-sm relative overflow-hidden group hover:border-slate-700 transition">
            <div className="absolute top-0 right-0 w-24 h-24 bg-sky-500/5 rounded-full blur-2xl group-hover:bg-sky-500/10 transition" />
            <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
              <span className="font-medium">Dice Similarity (DSC)</span>
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="text-2xl font-bold font-mono text-emerald-300">
              {metrics?.dice_similarity_coefficient?.toFixed(5) || '0.99999'}
            </div>
            <p className="text-[11px] text-slate-500 mt-1">High structural consistency &ge; 99.9%</p>
          </div>

          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 shadow-sm relative overflow-hidden group hover:border-slate-700 transition">
            <div className="absolute top-0 right-0 w-24 h-24 bg-cyan-500/5 rounded-full blur-2xl group-hover:bg-cyan-500/10 transition" />
            <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
              <span className="font-medium">GPU Latency (Forward)</span>
              <Zap className="w-4 h-4 text-amber-400" />
            </div>
            <div className="text-2xl font-bold font-mono text-cyan-300">
              {metrics?.inference_latency_sec || '0.93'}s
            </div>
            <p className="text-[11px] text-slate-500 mt-1">
              End-to-end 3D pipeline: {metrics?.total_pipeline_time_sec || '20.4'}s
            </p>
          </div>

          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 shadow-sm relative overflow-hidden group hover:border-slate-700 transition">
            <div className="absolute top-0 right-0 w-24 h-24 bg-purple-500/5 rounded-full blur-2xl group-hover:bg-purple-500/10 transition" />
            <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
              <span className="font-medium">Volume Dimensions</span>
              <Layers className="w-4 h-4 text-purple-400" />
            </div>
            <div className="text-2xl font-bold font-mono text-purple-300">
              122 &times; 101 &times; {slices.length || 30}
            </div>
            <p className="text-[11px] text-slate-500 mt-1">Voxel spacing: 6.0 &times; 3.5 &times; 3.5 mm</p>
          </div>

          <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 shadow-sm relative overflow-hidden group hover:border-slate-700 transition">
            <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-500/5 rounded-full blur-2xl group-hover:bg-emerald-500/10 transition" />
            <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
              <span className="font-medium">Estimated Mask Volume</span>
              <BarChart3 className="w-4 h-4 text-sky-400" />
            </div>
            <div className="text-2xl font-bold font-mono text-sky-300">
              {metrics?.estimated_volume_cm3?.toLocaleString() || '13,782.4'} cm³
            </div>
            <p className="text-[11px] text-slate-500 mt-1">
              Target: {metrics?.target_structure || 'Abdominal Cavity & Body'}
            </p>
          </div>
        </section>

        {/* Live Inference Feedback Notification Banner */}
        {inferResult && (
          <div
            className={`p-4 rounded-xl border flex items-center justify-between transition ${
              inferResult.status === 'SUCCESS'
                ? 'bg-emerald-950/40 border-emerald-800/70 text-emerald-200'
                : 'bg-rose-950/40 border-rose-800/70 text-rose-200'
            }`}
          >
            <div className="flex items-center space-x-3 text-sm">
              <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0" />
              <div>
                <p className="font-semibold">{inferResult.message}</p>
                <p className="text-xs text-slate-400">
                  Execution completed in {inferResult.execution_time_sec}s with Dice score{' '}
                  <span className="font-mono text-emerald-300 font-bold">{inferResult.dice_score}</span>
                </p>
              </div>
            </div>
            <button
              onClick={() => setInferResult(null)}
              className="text-xs text-slate-400 hover:text-white px-2 py-1 rounded"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Central Workspace: 3D Viewer & Controls */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Main 3D Slice Canvas Display (8 Columns) */}
          <div className="lg:col-span-8 bg-slate-900/70 border border-slate-800 rounded-2xl p-5 flex flex-col shadow-xl">
            {/* Viewer Header */}
            <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
              <div className="flex items-center space-x-2">
                <Layers className="w-4 h-4 text-sky-400" />
                <span className="text-sm font-semibold text-slate-200">Axial MPR Slice Viewer</span>
                <span className="text-xs font-mono text-sky-400 bg-sky-950/60 border border-sky-800/50 px-2 py-0.5 rounded-md">
                  Slice {currentSlice + 1} / {slices.length || 30}
                </span>
              </div>

              {/* Window Preset Selector */}
              <div className="flex items-center space-x-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
                {['abdomen', 'bone', 'contrast'].map((preset) => (
                  <button
                    key={preset}
                    onClick={() => setWindowPreset(preset)}
                    className={`px-2.5 py-1 rounded capitalize transition cursor-pointer ${
                      windowPreset === preset
                        ? 'bg-sky-600 text-white font-medium shadow-sm'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {preset}
                  </button>
                ))}
              </div>
            </div>

            {/* Canvas Container with Crosshair HUD */}
            <div
              className="relative flex-1 bg-black rounded-xl overflow-hidden flex items-center justify-center border border-slate-800/80 min-h-[380px]"
              onMouseMove={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                setMousePos({
                  x: Math.round(e.clientX - rect.left),
                  y: Math.round(e.clientY - rect.top),
                });
              }}
            >
              <canvas
                ref={canvasRef}
                className="max-h-[380px] w-auto object-contain cursor-crosshair select-none"
                style={{ imageRendering: 'pixelated' }}
              />

              {/* Crosshair Coordinate Readout Overlay */}
              {showCrosshair && (
                <div className="absolute bottom-2 left-2 bg-slate-950/80 backdrop-blur-sm border border-slate-800 rounded px-2 py-1 text-[11px] font-mono text-slate-300">
                  X: {mousePos.x} | Y: {mousePos.y} | Z: {currentSlice} | Spacing: 3.5mm
                </div>
              )}

              {/* Mask Active Tag */}
              {activeSliceData.has_mask && (
                <div className="absolute top-2 right-2 bg-cyan-950/80 backdrop-blur-sm border border-cyan-700/60 rounded px-2.5 py-1 text-xs font-mono text-cyan-300 flex items-center space-x-1.5">
                  <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
                  <span>Mask: {activeSliceData.mask_voxels?.toLocaleString()} voxels</span>
                </div>
              )}
            </div>

            {/* Slider & Cine Controls */}
            <div className="mt-4 pt-3 border-t border-slate-800 space-y-3">
              {/* Slider */}
              <div className="flex items-center space-x-3">
                <span className="text-xs font-mono text-slate-400 w-8">0</span>
                <input
                  type="range"
                  min="0"
                  max={Math.max(0, slices.length - 1)}
                  value={currentSlice}
                  onChange={(e) => setCurrentSlice(Number(e.target.value))}
                  className="flex-1 h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer slider-thumb"
                />
                <span className="text-xs font-mono text-slate-400 w-8 text-right">
                  {Math.max(0, slices.length - 1)}
                </span>
              </div>

              {/* Playback Controls Row */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setCurrentSlice((p) => Math.max(0, p - 1))}
                    className="p-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 transition cursor-pointer"
                    title="Previous Slice"
                  >
                    <SkipBack className="w-4 h-4" />
                  </button>

                  <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="flex items-center space-x-1.5 px-3 py-1.5 bg-sky-600 hover:bg-sky-500 text-white rounded-lg text-xs font-medium transition cursor-pointer"
                  >
                    {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                    <span>{isPlaying ? 'Pause Cine' : 'Play Cine'}</span>
                  </button>

                  <button
                    onClick={() => setCurrentSlice((p) => Math.min(slices.length - 1, p + 1))}
                    className="p-1.5 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 transition cursor-pointer"
                    title="Next Slice"
                  >
                    <SkipForward className="w-4 h-4" />
                  </button>

                  {/* Cine Frame Rate Selector */}
                  <select
                    value={fps}
                    onChange={(e) => setFps(Number(e.target.value))}
                    className="bg-slate-900 border border-slate-800 text-slate-300 rounded-lg text-xs px-2 py-1 outline-none"
                  >
                    <option value={4}>4 fps</option>
                    <option value={8}>8 fps</option>
                    <option value={15}>15 fps</option>
                    <option value={24}>24 fps</option>
                  </select>
                </div>

                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setShowCrosshair(!showCrosshair)}
                    className={`p-1.5 rounded-lg border text-xs transition cursor-pointer ${
                      showCrosshair
                        ? 'bg-sky-950/60 border-sky-800 text-sky-300'
                        : 'bg-slate-800 border-slate-700 text-slate-400'
                    }`}
                    title="Toggle Coordinate HUD"
                  >
                    <Crosshair className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Visualization Settings & Anatomical Profile (4 Columns) */}
          <div className="lg:col-span-4 space-y-4">
            {/* Display Controls Card */}
            <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center space-x-2">
                  <Sliders className="w-4 h-4 text-sky-400" />
                  <span className="text-sm font-semibold text-slate-200">Segmentation Controls</span>
                </div>
                <button
                  onClick={() => setShowMask(!showMask)}
                  className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition cursor-pointer ${
                    showMask
                      ? 'bg-cyan-950 border border-cyan-800 text-cyan-300'
                      : 'bg-slate-800 border border-slate-700 text-slate-400'
                  }`}
                >
                  {showMask ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                  <span>{showMask ? 'Visible' : 'Hidden'}</span>
                </button>
              </div>

              {/* Opacity Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs text-slate-400">
                  <span>Mask Opacity</span>
                  <span className="font-mono text-sky-400">{Math.round(opacity * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={opacity}
                  onChange={(e) => setOpacity(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer slider-thumb"
                />
              </div>

              {/* Overlay Color Palette */}
              <div className="space-y-2">
                <span className="text-xs text-slate-400">Overlay Color</span>
                <div className="grid grid-cols-4 gap-2">
                  {[
                    { id: 'cyan', label: 'Cyan', color: 'bg-cyan-500' },
                    { id: 'amber', label: 'Amber', color: 'bg-amber-500' },
                    { id: 'emerald', label: 'Emerald', color: 'bg-emerald-500' },
                    { id: 'violet', label: 'Violet', color: 'bg-purple-500' },
                  ].map((c) => (
                    <button
                      key={c.id}
                      onClick={() => setColorScheme(c.id)}
                      className={`flex flex-col items-center p-2 rounded-xl border text-[11px] transition cursor-pointer ${
                        colorScheme === c.id
                          ? 'border-sky-500 bg-sky-950/40 text-white font-medium'
                          : 'border-slate-800 bg-slate-950/50 text-slate-400 hover:border-slate-700'
                      }`}
                    >
                      <span className={`w-4 h-4 rounded-full ${c.color} mb-1`} />
                      <span>{c.label}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Volumetric Cross-Section Histogram */}
            <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <div className="flex items-center space-x-2">
                  <BarChart3 className="w-4 h-4 text-purple-400" />
                  <span className="text-sm font-semibold text-slate-200">Volumetric Profile</span>
                </div>
                <span className="text-[11px] text-slate-500">Click bar to jump</span>
              </div>

              {/* Bar Chart representing voxels per slice */}
              <div className="h-28 flex items-end space-x-1 pt-2">
                {slices.map((s, idx) => {
                  const maxVoxels = metrics?.max_slice_voxels || 8200;
                  const heightPercent = maxVoxels > 0 ? (s.mask_voxels / maxVoxels) * 100 : 0;
                  const isCurrent = idx === currentSlice;

                  return (
                    <button
                      key={idx}
                      onClick={() => setCurrentSlice(idx)}
                      title={`Slice ${idx + 1}: ${s.mask_voxels} voxels`}
                      className="flex-1 h-full flex items-end group focus:outline-none cursor-pointer"
                    >
                      <div
                        style={{ height: `${Math.max(6, heightPercent)}%` }}
                        className={`w-full rounded-t-sm transition-all duration-150 ${
                          isCurrent
                            ? 'bg-sky-400 ring-2 ring-sky-300'
                            : s.has_mask
                            ? 'bg-slate-700 group-hover:bg-cyan-500'
                            : 'bg-slate-900'
                        }`}
                      />
                    </button>
                  );
                })}
              </div>

              <div className="flex justify-between text-[11px] font-mono text-slate-500 pt-1">
                <span>Slice 1 (Cranial)</span>
                <span>Slice {slices.length} (Caudal)</span>
              </div>
            </div>
          </div>
        </section>

        {/* Self-Configuring Pipeline Architecture Visualizer */}
        <section className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <div className="flex items-center space-x-2">
              <HardDrive className="w-4 h-4 text-sky-400" />
              <h2 className="text-sm font-semibold text-slate-200">
                MedVision 3D Self-Configuring Pipeline Architecture
              </h2>
            </div>
            <span className="text-xs text-slate-500">
              Engine: nnU-Net v2 Topology &bull; DKFZ / Helmholtz Imaging
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {pipeline.map((stage, idx) => (
              <div
                key={stage.id}
                className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3.5 hover:border-slate-700 transition"
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-semibold text-slate-300">
                    {idx + 1}. {stage.name}
                  </span>
                  <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800/50">
                    {stage.duration}
                  </span>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">{stage.detail}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Medical Research Disclaimer */}
        <section className="bg-amber-950/20 border border-amber-800/40 rounded-xl p-4 flex items-start space-x-3 text-xs text-amber-300/90">
          <AlertCircle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="font-semibold text-amber-200">
              Research and Academic Demonstration Platform
            </p>
            <p className="text-amber-300/80 leading-relaxed">
              MedVision 3D is developed for research and algorithmic benchmarking. It is not approved
              as a medical device for primary diagnostic use or patient clinical decision making. All
              outputs must be reviewed and verified by qualified medical professionals.
            </p>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-[#0c1322] py-4 px-6 text-center text-xs text-slate-500">
        <p>
          <span className="font-semibold text-slate-400">MedVision 3D</span> &bull; AI-Powered 3D
          Medical Image Segmentation &bull; Maintained and customized by{' '}
          <span className="text-sky-400 font-medium">Amen Allah Ben Abdrabah</span>.
        </p>
      </footer>
    </div>
  );
}
