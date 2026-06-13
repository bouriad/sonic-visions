import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { AudioLines, Mic, Play, Shuffle, Sparkles } from "lucide-react";
import "./styles.css";

type ModeId =
  | "typograph"
  | "contours"
  | "lissajous"
  | "circuit"
  | "koi"
  | "seismo"
  | "shrine"
  | "rain"
  | "gravity"
  | "ink";

type Metrics = {
  rms: number;
  bass: number;
  mid: number;
  treble: number;
  centroid: number;
  pitch: number | null;
  note: string;
  noteIndex: number;
  onset: boolean;
  time: number;
};

type Particle = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  max: number;
  size: number;
  hue: number;
  text?: string;
  angle?: number;
  note?: string;
};

type DrawState = {
  particles: Particle[];
  lastMode: ModeId;
  demoPhase: number;
};

const MODES: Array<{ id: ModeId; name: string; tag: string }> = [
  { id: "typograph", name: "Type Weather", tag: "letters" },
  { id: "contours", name: "Velvet Topography", tag: "maps" },
  { id: "lissajous", name: "Lissajous Chapel", tag: "math" },
  { id: "circuit", name: "Circuit Choir", tag: "wires" },
  { id: "koi", name: "Koi Notation", tag: "ribbons" },
  { id: "seismo", name: "Brutalist Seismograph", tag: "paper" },
  { id: "shrine", name: "Glass Shrine", tag: "radial" },
  { id: "rain", name: "Pixel Rain Psalm", tag: "glyphs" },
  { id: "gravity", name: "Gravity Bloom", tag: "body" },
  { id: "ink", name: "Ink Score", tag: "wet" },
];

const NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
const WORDS = ["hush", "afterimage", "salt", "glass", "pulse", "murmur", "velvet", "static", "choir", "orbit", "ink", "minor"];

function clamp(value: number, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function noteFromPitch(freq: number | null) {
  if (!freq || !Number.isFinite(freq)) return { note: "—", index: 0 };
  const midi = Math.round(69 + 12 * Math.log2(freq / 440));
  const octave = Math.floor(midi / 12) - 1;
  const index = ((midi % 12) + 12) % 12;
  return { note: `${NOTE_NAMES[index]}${octave}`, index };
}

function autoCorrelate(buffer: Float32Array, sampleRate: number) {
  let rms = 0;
  for (let i = 0; i < buffer.length; i++) rms += buffer[i] * buffer[i];
  rms = Math.sqrt(rms / buffer.length);
  if (rms < 0.012) return null;

  let r1 = 0;
  let r2 = buffer.length - 1;
  const threshold = 0.18;
  for (let i = 0; i < buffer.length / 2; i++) {
    if (Math.abs(buffer[i]) < threshold) {
      r1 = i;
      break;
    }
  }
  for (let i = 1; i < buffer.length / 2; i++) {
    if (Math.abs(buffer[buffer.length - i]) < threshold) {
      r2 = buffer.length - i;
      break;
    }
  }

  const slice = buffer.slice(r1, r2);
  const correlations = new Array(slice.length).fill(0);
  for (let lag = 0; lag < slice.length; lag++) {
    for (let i = 0; i < slice.length - lag; i++) {
      correlations[lag] += slice[i] * slice[i + lag];
    }
  }

  let d = 0;
  while (correlations[d] > correlations[d + 1]) d++;
  let maxValue = -1;
  let maxIndex = -1;
  for (let i = d; i < correlations.length; i++) {
    if (correlations[i] > maxValue) {
      maxValue = correlations[i];
      maxIndex = i;
    }
  }
  if (maxIndex <= 0) return null;

  const x1 = correlations[maxIndex - 1] ?? 0;
  const x2 = correlations[maxIndex] ?? 0;
  const x3 = correlations[maxIndex + 1] ?? 0;
  const correction = (x1 - x3) / (2 * (x1 - 2 * x2 + x3));
  return sampleRate / (maxIndex + (Number.isFinite(correction) ? correction : 0));
}

function bandEnergy(freqData: Uint8Array, start: number, end: number) {
  let total = 0;
  let count = 0;
  for (let i = start; i < end && i < freqData.length; i++) {
    total += freqData[i] / 255;
    count++;
  }
  return count ? total / count : 0;
}

function makeSilentMetrics(time: number): Metrics {
  return {
    rms: 0,
    bass: 0,
    mid: 0,
    treble: 0,
    centroid: 0,
    pitch: null,
    note: "—",
    noteIndex: 0,
    onset: false,
    time,
  };
}

function drawBackground(ctx: CanvasRenderingContext2D, w: number, h: number, color = "#050505") {
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, w, h);
}

function spawn(state: DrawState, metrics: Metrics, w: number, h: number, count = 5) {
  for (let i = 0; i < count; i++) {
    const noteHue = (metrics.noteIndex * 31 + metrics.centroid * 120 + i * 16) % 360;
    state.particles.push({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * (1 + metrics.treble * 5),
      vy: (Math.random() - 0.5) * (1 + metrics.bass * 5),
      life: 1,
      max: 0.75 + Math.random() * 1.4,
      size: 10 + Math.random() * 42 + metrics.rms * 90,
      hue: noteHue,
      text: Math.random() > 0.35 ? metrics.note : WORDS[Math.floor(Math.random() * WORDS.length)],
      angle: (Math.random() - 0.5) * Math.PI,
      note: metrics.note,
    });
  }
}

function drawParticles(ctx: CanvasRenderingContext2D, state: DrawState, dt: number, style: "text" | "dot" | "ink") {
  state.particles = state.particles.filter((p) => {
    p.life -= dt / p.max;
    p.x += p.vx * 60 * dt;
    p.y += p.vy * 60 * dt;
    if (p.life <= 0) return false;
    ctx.save();
    ctx.globalAlpha = Math.max(0, p.life);
    if (style === "text") {
      ctx.translate(p.x, p.y);
      ctx.rotate(p.angle ?? 0);
      ctx.font = `${Math.max(14, p.size)}px ${Math.random() > 0.5 ? "serif" : "monospace"}`;
      ctx.fillStyle = `hsl(${p.hue} 85% ${style === "text" ? "12%" : "62%"})`;
      ctx.fillText(p.text ?? "", 0, 0);
    } else if (style === "ink") {
      const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size * 1.4);
      grad.addColorStop(0, `hsla(${p.hue}, 85%, 60%, ${0.45 * p.life})`);
      grad.addColorStop(1, `hsla(${p.hue}, 85%, 30%, 0)`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * (1.15 - p.life * 0.2), 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.fillStyle = `hsla(${p.hue}, 90%, 64%, ${p.life})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * 0.12, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
    return true;
  });
}

function drawMode(ctx: CanvasRenderingContext2D, mode: ModeId, metrics: Metrics, waveform: Uint8Array, freqData: Uint8Array, state: DrawState, dt: number) {
  const ratio = window.devicePixelRatio || 1;
  const w = ctx.canvas.width / ratio;
  const h = ctx.canvas.height / ratio;
  if (state.lastMode !== mode) {
    state.particles = [];
    state.lastMode = mode;
  }
  if (metrics.onset) spawn(state, metrics, w, h, mode === "typograph" ? 8 : 4);

  if (mode === "typograph") {
    drawBackground(ctx, w, h, "#f8f6ef");
    drawParticles(ctx, state, dt, "text");
    return;
  }

  if (mode === "contours") {
    drawBackground(ctx, w, h, "#080607");
    const step = Math.max(18, Math.floor(w / 22));
    for (let y = -step; y < h + step; y += step) {
      ctx.beginPath();
      for (let x = -step; x < w + step; x += 8) {
        const n = Math.sin(x * 0.015 + metrics.time * 1.8) + Math.cos(y * 0.02 + metrics.noteIndex);
        const yy = y + n * (18 + metrics.mid * 90) + Math.sin((x + y) * 0.008) * metrics.bass * 60;
        if (x === -step) ctx.moveTo(x, yy);
        else ctx.lineTo(x, yy);
      }
      ctx.strokeStyle = `hsla(${260 + metrics.noteIndex * 12}, 70%, ${42 + metrics.treble * 30}%, .38)`;
      ctx.lineWidth = 1 + metrics.rms * 5;
      ctx.stroke();
    }
    drawParticles(ctx, state, dt, "ink");
    return;
  }

  if (mode === "lissajous") {
    drawBackground(ctx, w, h, "#03060a");
    ctx.save();
    ctx.translate(w / 2, h / 2);
    for (let ring = 0; ring < 9; ring++) {
      ctx.beginPath();
      const a = 2 + (metrics.noteIndex % 5);
      const b = 3 + ((metrics.noteIndex + ring) % 7);
      const radius = Math.min(w, h) * (0.12 + ring * 0.035 + metrics.rms * 0.08);
      for (let i = 0; i <= 520; i++) {
        const t = (i / 520) * Math.PI * 2;
        const x = Math.sin(a * t + metrics.time * 0.5) * radius * (1 + metrics.bass * 0.6);
        const y = Math.sin(b * t + ring + metrics.time * 0.3) * radius * (1 + metrics.treble * 0.5);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = `hsla(${180 + ring * 16 + metrics.noteIndex * 8}, 90%, 64%, ${0.18 + ring * 0.045})`;
      ctx.lineWidth = 0.8 + metrics.mid * 4;
      ctx.stroke();
    }
    ctx.restore();
    return;
  }

  if (mode === "circuit") {
    drawBackground(ctx, w, h, "#050805");
    const cols = 7;
    const rows = 12;
    for (let y = 0; y <= rows; y++) {
      for (let x = 0; x <= cols; x++) {
        const px = (x / cols) * w;
        const py = (y / rows) * h;
        const pulse = Math.sin(metrics.time * 5 + x + y + metrics.noteIndex) * 0.5 + 0.5;
        ctx.fillStyle = `hsla(${100 + metrics.noteIndex * 18}, 90%, ${20 + pulse * 45}%, ${0.16 + metrics.rms})`;
        ctx.fillRect(px - 2, py - 2, 4 + metrics.treble * 8, 4 + metrics.treble * 8);
        if (x < cols) {
          ctx.strokeStyle = `hsla(${150 + metrics.noteIndex * 12}, 80%, 55%, ${0.07 + pulse * 0.18})`;
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(px + w / cols, py + Math.sin(metrics.time + x * y) * metrics.mid * 16);
          ctx.stroke();
        }
      }
    }
    drawParticles(ctx, state, dt, "dot");
    return;
  }

  if (mode === "koi") {
    drawBackground(ctx, w, h, "#06100f");
    for (let ribbon = 0; ribbon < 11; ribbon++) {
      ctx.beginPath();
      for (let i = 0; i < 180; i++) {
        const t = i / 179;
        const x = t * w;
        const y = h * (0.18 + ribbon * 0.07) + Math.sin(t * 8 + metrics.time * (0.7 + ribbon * 0.03) + ribbon) * (18 + metrics.bass * 90);
        const sway = Math.sin(t * 18 + metrics.noteIndex) * metrics.treble * 55;
        if (i === 0) ctx.moveTo(x, y + sway);
        else ctx.lineTo(x, y + sway);
      }
      ctx.strokeStyle = `hsla(${22 + ribbon * 19 + metrics.noteIndex * 4}, 95%, 62%, ${0.22 + metrics.mid * 0.3})`;
      ctx.lineWidth = 2 + metrics.rms * 18;
      ctx.stroke();
    }
    return;
  }

  if (mode === "seismo") {
    drawBackground(ctx, w, h, "#f4f0e7");
    const rows = 13;
    for (let r = 0; r < rows; r++) {
      const y = (r + 1) * (h / (rows + 1));
      ctx.beginPath();
      for (let i = 0; i < waveform.length; i += 4) {
        const x = (i / waveform.length) * w;
        const v = (waveform[i] - 128) / 128;
        const yy = y + v * (8 + metrics.rms * 90) + Math.sin(i * 0.03 + r) * metrics.bass * 18;
        if (i === 0) ctx.moveTo(x, yy);
        else ctx.lineTo(x, yy);
      }
      ctx.strokeStyle = r % 3 === 0 ? "#111" : `hsla(${metrics.noteIndex * 30}, 75%, 35%, .42)`;
      ctx.lineWidth = r % 3 === 0 ? 1.5 : 0.8;
      ctx.stroke();
    }
    return;
  }

  if (mode === "shrine") {
    drawBackground(ctx, w, h, "#08040a");
    ctx.save();
    ctx.translate(w / 2, h / 2);
    const sides = 10 + (metrics.noteIndex % 5);
    for (let layer = 0; layer < 18; layer++) {
      const radius = Math.min(w, h) * (0.06 + layer * 0.025 + metrics.rms * 0.08);
      ctx.beginPath();
      for (let i = 0; i <= sides; i++) {
        const a = (i / sides) * Math.PI * 2 + metrics.time * 0.08 * (layer % 2 ? 1 : -1);
        const rr = radius * (1 + Math.sin(a * 3 + metrics.time + layer) * metrics.mid * 0.24);
        const x = Math.cos(a) * rr;
        const y = Math.sin(a) * rr;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.fillStyle = `hsla(${layer * 18 + metrics.noteIndex * 28}, 88%, ${36 + metrics.treble * 28}%, .09)`;
      ctx.strokeStyle = `hsla(${layer * 18 + metrics.noteIndex * 28}, 88%, 68%, .28)`;
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
    return;
  }

  if (mode === "rain") {
    drawBackground(ctx, w, h, "#020205");
    const glyphs = "01アイウエオ音階光雨";
    const cols = Math.floor(w / 18);
    ctx.font = "15px monospace";
    for (let c = 0; c < cols; c++) {
      const speed = 20 + ((c * 13 + metrics.noteIndex * 9) % 80) + metrics.treble * 220;
      const y = (metrics.time * speed + c * 37) % (h + 80);
      for (let k = 0; k < 12; k++) {
        const char = glyphs[(c + k + metrics.noteIndex) % glyphs.length];
        ctx.fillStyle = `hsla(${145 + metrics.noteIndex * 16}, 95%, ${25 + k * 4}%, ${0.12 + metrics.rms * 0.7})`;
        ctx.fillText(char, c * 18, y - k * 22);
      }
    }
    return;
  }

  if (mode === "gravity") {
    drawBackground(ctx, w, h, "#03030a");
    const cx = w / 2 + Math.sin(metrics.time * 0.9) * metrics.mid * w * 0.2;
    const cy = h / 2 + Math.cos(metrics.time * 0.7) * metrics.bass * h * 0.18;
    for (let i = 0; i < 120; i++) {
      const a = i * 2.399 + metrics.time * (0.12 + metrics.treble * 0.4);
      const r = Math.sqrt(i) * (8 + metrics.rms * 18);
      const x = cx + Math.cos(a) * r;
      const y = cy + Math.sin(a) * r;
      ctx.fillStyle = `hsla(${metrics.noteIndex * 24 + i}, 90%, ${40 + metrics.mid * 38}%, ${0.12 + i / 600})`;
      ctx.beginPath();
      ctx.arc(x, y, 1.6 + metrics.bass * 8, 0, Math.PI * 2);
      ctx.fill();
    }
    drawParticles(ctx, state, dt, "dot");
    return;
  }

  drawBackground(ctx, w, h, "#f7f1e7");
  if (mode === "ink") {
    if (metrics.onset) spawn(state, metrics, w, h, 10);
    drawParticles(ctx, state, dt, "ink");
    ctx.strokeStyle = "rgba(20, 12, 10, .55)";
    ctx.lineWidth = 1 + metrics.rms * 6;
    ctx.beginPath();
    for (let i = 0; i < waveform.length; i += 3) {
      const x = (i / waveform.length) * w;
      const y = h * 0.52 + ((waveform[i] - 128) / 128) * (60 + metrics.rms * 140);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

function useAudio() {
  const [status, setStatus] = useState<"idle" | "mic" | "demo" | "error">("idle");
  const [error, setError] = useState("");
  const ctxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const lastRmsRef = useRef(0);

  async function startMic() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.72;
      source.connect(analyser);
      ctxRef.current = audioCtx;
      analyserRef.current = analyser;
      streamRef.current = stream;
      setStatus("mic");
      setError("");
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Mic unavailable");
    }
  }

  function startDemo() {
    stop();
    setStatus("demo");
    setError("");
  }

  function stop() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    void ctxRef.current?.close();
    ctxRef.current = null;
    analyserRef.current = null;
    setStatus("idle");
  }

  function read(time: number): { metrics: Metrics; waveform: Uint8Array; freqData: Uint8Array } {
    const waveform = new Uint8Array(2048);
    const freqData = new Uint8Array(1024);
    const floatWave = new Float32Array(2048);
    const analyser = analyserRef.current;
    const ctx = ctxRef.current;

    if (status === "demo") {
      const phase = time * 0.85;
      for (let i = 0; i < waveform.length; i++) {
        const t = i / waveform.length;
        waveform[i] = 128 + Math.sin(t * Math.PI * 16 + phase * 4) * 55 + Math.sin(t * Math.PI * 31 + phase) * 24;
      }
      for (let i = 0; i < freqData.length; i++) {
        freqData[i] = Math.max(0, 180 * Math.exp(-i / 96) * (0.65 + 0.35 * Math.sin(phase + i * 0.04)));
      }
      const fakePitch = [220, 246.94, 261.63, 329.63, 392, 440][Math.floor(phase * 2) % 6];
      const note = noteFromPitch(fakePitch);
      const rms = 0.18 + 0.14 * (Math.sin(phase * 3) * 0.5 + 0.5);
      const onset = Math.abs(Math.sin(phase * Math.PI)) > 0.94;
      return {
        waveform,
        freqData,
        metrics: { rms, bass: 0.48, mid: 0.38, treble: 0.32, centroid: 0.3, pitch: fakePitch, note: note.note, noteIndex: note.index, onset, time },
      };
    }

    if (!analyser || !ctx) {
      return { metrics: makeSilentMetrics(time), waveform, freqData };
    }

    analyser.getByteTimeDomainData(waveform);
    analyser.getByteFrequencyData(freqData);
    analyser.getFloatTimeDomainData(floatWave);
    const pitch = autoCorrelate(floatWave, ctx.sampleRate);
    const note = noteFromPitch(pitch);
    let rms = 0;
    for (let i = 0; i < floatWave.length; i++) rms += floatWave[i] * floatWave[i];
    rms = Math.sqrt(rms / floatWave.length);
    const bass = bandEnergy(freqData, 2, 12);
    const mid = bandEnergy(freqData, 12, 90);
    const treble = bandEnergy(freqData, 90, 260);
    let weighted = 0;
    let total = 0;
    for (let i = 0; i < freqData.length; i++) {
      const value = freqData[i] / 255;
      weighted += i * value;
      total += value;
    }
    const centroid = total ? weighted / (freqData.length * total) : 0;
    const onset = rms > 0.025 && rms > lastRmsRef.current * 1.45;
    lastRmsRef.current = lerp(lastRmsRef.current, rms, 0.45);
    return { waveform, freqData, metrics: { rms: clamp(rms * 4), bass, mid, treble, centroid, pitch, note: note.note, noteIndex: note.index, onset, time } };
  }

  return { status, error, startMic, startDemo, stop, read };
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stateRef = useRef<DrawState>({ particles: [], lastMode: "typograph", demoPhase: 0 });
  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef(0);
  const audioRef = useRef<ReturnType<typeof useAudio> | null>(null);
  const [mode, setMode] = useState<ModeId>("typograph");
  const [metrics, setMetrics] = useState<Metrics>(makeSilentMetrics(0));
  const audio = useAudio();
  audioRef.current = audio;

  const activeMode = useMemo(() => MODES.find((item) => item.id === mode) ?? MODES[0], [mode]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const requestedMode = params.get("mode") as ModeId | null;
    if (requestedMode && MODES.some((item) => item.id === requestedMode)) {
      setMode(requestedMode);
    }
    if (params.get("demo") === "1") {
      audio.startDemo();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.floor(window.innerWidth * ratio);
      canvas.height = Math.floor(window.innerHeight * ratio);
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const tick = (now: number) => {
      const seconds = now / 1000;
      const dt = Math.min(0.05, seconds - (lastTimeRef.current || seconds));
      lastTimeRef.current = seconds;
      const packet = audioRef.current?.read(seconds) ?? {
        metrics: makeSilentMetrics(seconds),
        waveform: new Uint8Array(2048),
        freqData: new Uint8Array(1024),
      };
      setMetrics(packet.metrics);
      drawMode(ctx, mode, packet.metrics, packet.waveform, packet.freqData, stateRef.current, dt);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      window.removeEventListener("resize", resize);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [mode]);

  return (
    <main className="app-shell">
      <canvas ref={canvasRef} className="visual-canvas" />

      <section className="top-panel">
        <div>
          <p className="eyebrow">Sonic Visions</p>
          <h1>{activeMode.name}</h1>
        </div>
        <div className="note-chip">
          <span>{metrics.note}</span>
          <small>{Math.round(metrics.rms * 100)}%</small>
        </div>
      </section>

      <section className="meter-panel">
        <div className="meter"><span style={{ width: `${metrics.bass * 100}%` }} /></div>
        <div className="meter"><span style={{ width: `${metrics.mid * 100}%` }} /></div>
        <div className="meter"><span style={{ width: `${metrics.treble * 100}%` }} /></div>
      </section>

      <section className="controls">
        <div className="mode-strip">
          {MODES.map((item) => (
            <button key={item.id} className={item.id === mode ? "selected" : ""} onClick={() => setMode(item.id)}>
              <span>{item.name}</span>
              <small>{item.tag}</small>
            </button>
          ))}
        </div>
        <div className="action-row">
          <button className="action primary" onClick={audio.startMic}>
            <Mic size={18} />
            Mic
          </button>
          <button className="action" onClick={audio.startDemo}>
            <Play size={18} />
            Demo
          </button>
          <button className="action" onClick={() => setMode(MODES[Math.floor(Math.random() * MODES.length)].id)}>
            <Shuffle size={18} />
            Shuffle
          </button>
        </div>
      </section>

      <section className="status-pill">
        {audio.status === "mic" ? <AudioLines size={15} /> : <Sparkles size={15} />}
        <span>{audio.status === "error" ? audio.error : audio.status}</span>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
