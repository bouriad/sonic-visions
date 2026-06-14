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
  notes: DetectedNote[];
  chord: string;
  onset: boolean;
  time: number;
};

type DetectedNote = {
  note: string;
  index: number;
  frequency: number;
  strength: number;
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
  { id: "typograph", name: "Silent Room", tag: "minimal" },
  { id: "lissajous", name: "Whitney Chords", tag: "harmony" },
  { id: "contours", name: "Reaction Bloom", tag: "morphogens" },
  { id: "koi", name: "Spectral Loom", tag: "woven chord" },
  { id: "rain", name: "Cellular Storm", tag: "automata" },
  { id: "circuit", name: "Signal City", tag: "brutal grid" },
  { id: "ink", name: "Black Mass", tag: "chaos ink" },
];

const NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
const WORDS = ["hush", "afterimage", "salt", "glass", "pulse", "murmur", "velvet", "static", "choir", "orbit", "ink", "minor"];
const GLYPHS = "01アイウエオ音階光雨░▒▓◌◍◇◆∿∴∵";

function clamp(value: number, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function noteFromPitch(freq: number | null) {
  if (!freq || !Number.isFinite(freq)) return { note: "—", index: 0 };
  return notePartsFromFrequency(freq);
}

function notePartsFromFrequency(freq: number) {
  const midi = Math.round(69 + 12 * Math.log2(freq / 440));
  const octave = Math.floor(midi / 12) - 1;
  const index = ((midi % 12) + 12) % 12;
  return { note: `${NOTE_NAMES[index]}${octave}`, index };
}

function detectSpectrumNotes(freqData: Uint8Array, sampleRate: number, fftSize: number): DetectedNote[] {
  const nyquist = sampleRate / 2;
  const minBin = Math.max(2, Math.floor((70 / nyquist) * freqData.length));
  const maxBin = Math.min(freqData.length - 2, Math.ceil((1800 / nyquist) * freqData.length));
  let maxValue = 0;
  let sum = 0;
  let count = 0;
  for (let i = minBin; i <= maxBin; i++) {
    const value = freqData[i] / 255;
    maxValue = Math.max(maxValue, value);
    sum += value;
    count++;
  }
  const avg = count ? sum / count : 0;
  const threshold = Math.max(0.08, avg * 2.2, maxValue * 0.28);
  const peaks: DetectedNote[] = [];

  for (let i = minBin; i <= maxBin; i++) {
    const value = freqData[i] / 255;
    if (value < threshold || value < freqData[i - 1] / 255 || value < freqData[i + 1] / 255) continue;
    const frequency = (i * sampleRate) / fftSize;
    const parts = notePartsFromFrequency(frequency);
    peaks.push({ ...parts, frequency, strength: value });
  }

  peaks.sort((a, b) => b.strength - a.strength);
  const byPitchClass = new Map<number, DetectedNote>();
  for (const peak of peaks) {
    const existing = byPitchClass.get(peak.index);
    if (!existing || peak.strength > existing.strength) byPitchClass.set(peak.index, peak);
    if (byPitchClass.size >= 7) break;
  }

  return [...byPitchClass.values()].sort((a, b) => b.strength - a.strength).slice(0, 6);
}

function labelChord(notes: DetectedNote[], fallback: string) {
  if (notes.length >= 2) return notes.map((n) => n.note.replace(/\d/g, "")).join(" ");
  return fallback;
}

function autoCorrelate(buffer: Float32Array, sampleRate: number) {
  let rms = 0;
  for (let i = 0; i < buffer.length; i++) rms += buffer[i] * buffer[i];
  rms = Math.sqrt(rms / buffer.length);
  if (rms < 0.004) return null;

  let r1 = 0;
  let r2 = buffer.length - 1;
  const threshold = 0.08;
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

function liftedFallbackStrength(buffer: Float32Array) {
  let rms = 0;
  for (let i = 0; i < buffer.length; i++) rms += buffer[i] * buffer[i];
  return clamp(Math.sqrt(rms / buffer.length) * 8);
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
    notes: [],
    chord: "—",
    onset: false,
    time,
  };
}

function drawBackground(ctx: CanvasRenderingContext2D, w: number, h: number, color = "#050505") {
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, w, h);
}

function grain(ctx: CanvasRenderingContext2D, w: number, h: number, alpha = 0.06) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = "#fff";
  for (let i = 0; i < 900; i++) {
    const x = (Math.sin(i * 12.9898) * 43758.5453) % 1;
    const y = (Math.sin(i * 78.233) * 24634.6345) % 1;
    ctx.fillRect(Math.abs(x) * w, Math.abs(y) * h, 1, 1);
  }
  ctx.restore();
}

function vignette(ctx: CanvasRenderingContext2D, w: number, h: number, strength = 0.72) {
  const g = ctx.createRadialGradient(w / 2, h / 2, Math.min(w, h) * 0.12, w / 2, h / 2, Math.max(w, h) * 0.72);
  g.addColorStop(0, "rgba(0,0,0,0)");
  g.addColorStop(1, `rgba(0,0,0,${strength})`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
}

function glowLine(ctx: CanvasRenderingContext2D, color: string, blur: number, width: number) {
  ctx.strokeStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = blur;
  ctx.lineWidth = width;
}

function waveAt(waveform: Uint8Array, index: number) {
  return ((waveform[index % waveform.length] ?? 128) - 128) / 128;
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
      text: Math.random() > 0.35 ? metrics.chord : WORDS[Math.floor(Math.random() * WORDS.length)],
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
    drawBackground(ctx, w, h, "#f7f4ea");
    if (metrics.rms < 0.025) {
      ctx.fillStyle = "rgba(10, 10, 9, .34)";
      ctx.fillRect(w * 0.5 - 0.5, h * 0.5 - 22, 1, 44);
      return;
    }
    const cx = w * 0.5;
    const cy = h * 0.5;
    ctx.strokeStyle = "rgba(9, 9, 8, .82)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, h * 0.12);
    ctx.lineTo(cx, h * 0.88);
    ctx.stroke();
    metrics.notes.forEach((note, i) => {
      const angle = -Math.PI / 2 + (i / Math.max(1, metrics.notes.length - 1)) * Math.PI;
      const radius = Math.min(w, h) * (0.08 + note.strength * 0.28);
      const x = cx + Math.cos(angle) * radius;
      const y = cy + Math.sin(angle) * radius;
      ctx.fillStyle = `hsla(${note.index * 30}, 75%, 38%, ${0.28 + note.strength * 0.55})`;
      ctx.beginPath();
      ctx.arc(x, y, 2 + note.strength * 16, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillStyle = "rgba(9,9,8,.58)";
      ctx.fillText(note.note, x + 12, y + 4);
    });
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(Math.sin(metrics.time * 0.5) * 0.02);
    ctx.font = `${10 + metrics.rms * 18}px ui-monospace, monospace`;
    ctx.fillStyle = "rgba(9,9,8,.56)";
    ctx.textAlign = "center";
    ctx.fillText(metrics.chord, 0, 58 + metrics.rms * 28);
    ctx.restore();
    return;
  }

  if (mode === "contours") {
    drawBackground(ctx, w, h, "#090607");
    for (let i = 0; i < 8; i++) {
      const x = w * (0.12 + i * 0.11 + Math.sin(metrics.time * 0.2 + i) * 0.02);
      const y = h * (0.18 + Math.cos(metrics.time * 0.22 + i) * 0.18 + (i % 3) * 0.2);
      const r = Math.min(w, h) * (0.12 + metrics.bass * 0.22 + i * 0.015);
      const g = ctx.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0, `hsla(${300 + metrics.noteIndex * 12 + i * 18}, 80%, 52%, .34)`);
      g.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);
    }
    const step = Math.max(14, Math.floor(w / 30));
    for (let y = -step; y < h + step; y += step) {
      ctx.beginPath();
      for (let x = -step; x < w + step; x += 6) {
        const n = Math.sin(x * 0.014 + metrics.time * 1.6) + Math.cos(y * 0.018 + metrics.noteIndex);
        const yy = y + n * (12 + metrics.mid * 110) + Math.sin((x + y) * 0.008 + metrics.time) * metrics.bass * 70;
        if (x === -step) ctx.moveTo(x, yy);
        else ctx.lineTo(x, yy);
      }
      glowLine(ctx, `hsla(${250 + metrics.noteIndex * 13}, 90%, ${55 + metrics.treble * 24}%, .44)`, 8, 0.7 + metrics.rms * 4);
      ctx.stroke();
    }
    ctx.shadowBlur = 0;
    drawParticles(ctx, state, dt, "ink");
    vignette(ctx, w, h, 0.62);
    return;
  }

  if (mode === "lissajous") {
    drawBackground(ctx, w, h, "#02040a");
    ctx.fillStyle = "rgba(210, 245, 255, .18)";
    for (let i = 0; i < 180; i++) {
      const x = Math.abs((Math.sin(i * 18.17 + metrics.noteIndex) * 9999) % 1) * w;
      const y = Math.abs((Math.cos(i * 11.73 + metrics.time * 0.12) * 9999) % 1) * h;
      const s = i % 9 === 0 ? 1.6 + metrics.treble * 3 : 0.8;
      ctx.fillRect(x, y, s, s);
    }
    ctx.save();
    ctx.translate(w / 2, h / 2);
    ctx.globalCompositeOperation = "lighter";
    const orbitNotes = metrics.notes.length ? metrics.notes : [{ note: metrics.note, index: metrics.noteIndex, frequency: metrics.pitch ?? 220, strength: metrics.rms }];
    for (let ring = 0; ring < 14; ring++) {
      const harmonic = orbitNotes[ring % orbitNotes.length];
      ctx.beginPath();
      const a = 2 + (harmonic.index % 5);
      const b = 3 + ((harmonic.index + ring) % 7);
      const radius = Math.min(w, h) * (0.075 + ring * 0.031 + harmonic.strength * 0.12);
      for (let i = 0; i <= 640; i++) {
        const t = (i / 640) * Math.PI * 2;
        const x = Math.sin(a * t + metrics.time * (0.35 + harmonic.strength * 0.22)) * radius * (1 + metrics.bass * 0.6);
        const y = Math.sin(b * t + ring + metrics.time * 0.28) * radius * (1 + metrics.treble * 0.5);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      glowLine(ctx, `hsla(${170 + ring * 18 + harmonic.index * 13}, 95%, 64%, ${0.11 + harmonic.strength * 0.28})`, 16, 0.6 + harmonic.strength * 4);
      ctx.stroke();
    }
    ctx.restore();
    ctx.shadowBlur = 0;
    vignette(ctx, w, h, 0.55);
    return;
  }

  if (mode === "circuit") {
    drawBackground(ctx, w, h, "#06100b");
    const cols = 9;
    const rows = 14;
    ctx.lineCap = "square";
    for (let y = 0; y <= rows; y++) {
      for (let x = 0; x <= cols; x++) {
        const px = (x / cols) * w + Math.sin(y + metrics.time) * metrics.mid * 8;
        const py = (y / rows) * h + Math.cos(x + metrics.time) * metrics.bass * 8;
        const pulse = Math.sin(metrics.time * 5 + x + y + metrics.noteIndex) * 0.5 + 0.5;
        ctx.fillStyle = `hsla(${95 + metrics.noteIndex * 18}, 95%, ${26 + pulse * 48}%, ${0.22 + metrics.rms})`;
        ctx.fillRect(px - 2 - metrics.treble * 3, py - 2 - metrics.treble * 3, 4 + metrics.treble * 10, 4 + metrics.treble * 10);
        if (x < cols) {
          glowLine(ctx, `hsla(${145 + metrics.noteIndex * 12}, 85%, 55%, ${0.08 + pulse * 0.28})`, 7, 1 + metrics.rms * 3);
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(px + w / cols, py);
          if ((x + y + metrics.noteIndex) % 3 === 0) ctx.lineTo(px + w / cols, py + h / rows);
          ctx.stroke();
        }
      }
    }
    ctx.shadowBlur = 0;
    drawParticles(ctx, state, dt, "dot");
    vignette(ctx, w, h, 0.5);
    return;
  }

  if (mode === "koi") {
    drawBackground(ctx, w, h, "#06100f");
    const water = ctx.createLinearGradient(0, 0, w, h);
    water.addColorStop(0, "rgba(35, 84, 80, .24)");
    water.addColorStop(1, "rgba(130, 42, 31, .18)");
    ctx.fillStyle = water;
    ctx.fillRect(0, 0, w, h);
    const threads = metrics.notes.length ? metrics.notes : [{ note: metrics.note, index: metrics.noteIndex, frequency: metrics.pitch ?? 220, strength: metrics.rms }];
    for (let ribbon = 0; ribbon < 18; ribbon++) {
      const thread = threads[ribbon % threads.length];
      ctx.beginPath();
      for (let i = 0; i < 180; i++) {
        const t = i / 179;
        const x = t * w;
        const y = h * (0.1 + ribbon * 0.05) + Math.sin(t * (6 + thread.index * 0.4) + metrics.time * (0.5 + thread.strength * 0.6) + ribbon) * (14 + metrics.bass * 120);
        const sway = Math.sin(t * 20 + thread.index + ribbon) * metrics.treble * 70;
        if (i === 0) ctx.moveTo(x, y + sway);
        else ctx.lineTo(x, y + sway);
      }
      glowLine(ctx, `hsla(${18 + ribbon * 13 + thread.index * 22}, 95%, 62%, ${0.14 + thread.strength * 0.42})`, 10, 1.5 + thread.strength * 18);
      ctx.stroke();
      for (let dot = 0; dot < 7; dot++) {
        const t = (dot + 1) / 8;
        const x = t * w;
        const y = h * (0.1 + ribbon * 0.05) + Math.sin(t * (6 + thread.index * 0.4) + metrics.time * (0.5 + thread.strength * 0.6) + ribbon) * (14 + metrics.bass * 120);
        ctx.fillStyle = `hsla(${40 + thread.index * 24}, 90%, 70%, .18)`;
        ctx.beginPath();
        ctx.ellipse(x, y, 3 + thread.strength * 18, 1.5 + metrics.treble * 6, 0, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.shadowBlur = 0;
    vignette(ctx, w, h, 0.42);
    return;
  }

  if (mode === "seismo") {
    drawBackground(ctx, w, h, "#eee6d2");
    grain(ctx, w, h, 0.11);
    ctx.fillStyle = "rgba(20, 17, 13, .08)";
    for (let x = 0; x < w; x += 38) ctx.fillRect(x, 0, 1, h);
    for (let y = 0; y < h; y += 38) ctx.fillRect(0, y, w, 1);
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
      if (r % 4 === 0) {
        ctx.font = "10px ui-monospace, monospace";
        ctx.fillStyle = "rgba(20,17,13,.52)";
        ctx.fillText(`${metrics.note} / ${Math.round(metrics.rms * 100).toString().padStart(2, "0")}`, 14, y - 6);
      }
    }
    return;
  }

  if (mode === "shrine") {
    drawBackground(ctx, w, h, "#07030a");
    ctx.save();
    ctx.translate(w / 2, h / 2);
    const sides = 10 + (metrics.noteIndex % 5);
    ctx.globalCompositeOperation = "lighter";
    for (let layer = 0; layer < 24; layer++) {
      const radius = Math.min(w, h) * (0.05 + layer * 0.022 + metrics.rms * 0.1);
      ctx.beginPath();
      for (let i = 0; i <= sides; i++) {
        const a = (i / sides) * Math.PI * 2 + metrics.time * 0.08 * (layer % 2 ? 1 : -1);
        const rr = radius * (1 + Math.sin(a * 3 + metrics.time + layer) * metrics.mid * 0.24);
        const x = Math.cos(a) * rr;
        const y = Math.sin(a) * rr;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.fillStyle = `hsla(${layer * 17 + metrics.noteIndex * 28}, 88%, ${38 + metrics.treble * 28}%, .08)`;
      glowLine(ctx, `hsla(${layer * 17 + metrics.noteIndex * 28}, 88%, 68%, .24)`, 9, 0.8 + metrics.rms * 3);
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
    ctx.shadowBlur = 0;
    vignette(ctx, w, h, 0.54);
    return;
  }

  if (mode === "rain") {
    drawBackground(ctx, w, h, "#020205");
    const mist = ctx.createLinearGradient(0, 0, w, h);
    mist.addColorStop(0, "rgba(13, 80, 58, .16)");
    mist.addColorStop(0.45, "rgba(23, 12, 70, .12)");
    mist.addColorStop(1, "rgba(110, 18, 48, .10)");
    ctx.fillStyle = mist;
    ctx.fillRect(0, 0, w, h);
    const cols = Math.floor(w / 14);
    ctx.font = "16px ui-monospace, monospace";
    for (let c = 0; c < cols; c++) {
      const speed = 20 + ((c * 13 + metrics.noteIndex * 9) % 80) + metrics.treble * 220;
      const y = (metrics.time * speed + c * 37) % (h + 80);
      for (let k = 0; k < 18; k++) {
        const char = GLYPHS[(c * 3 + k + metrics.noteIndex) % GLYPHS.length];
        const hot = k < 2 || metrics.onset;
        ctx.fillStyle = hot ? "#f5f1e8" : `hsla(${130 + metrics.noteIndex * 17}, 95%, ${28 + k * 3}%, ${0.18 + metrics.rms * 0.72})`;
        ctx.fillText(char, c * 14 + Math.sin(k + metrics.time) * metrics.bass * 6, y - k * 20);
      }
    }
    vignette(ctx, w, h, 0.65);
    return;
  }

  if (mode === "gravity") {
    drawBackground(ctx, w, h, "#03030a");
    const field = ctx.createRadialGradient(w * 0.5, h * 0.5, 0, w * 0.5, h * 0.5, Math.max(w, h) * 0.75);
    field.addColorStop(0, `hsla(${250 + metrics.noteIndex * 10}, 70%, 28%, .16)`);
    field.addColorStop(0.55, "rgba(22, 52, 78, .14)");
    field.addColorStop(1, "rgba(95, 18, 54, .10)");
    ctx.fillStyle = field;
    ctx.fillRect(0, 0, w, h);
    for (let lane = 0; lane < 16; lane++) {
      ctx.beginPath();
      for (let i = 0; i <= 160; i++) {
        const t = i / 160;
        const x = t * w;
        const y = h * ((lane + 0.5) / 16) + Math.sin(t * 8 + lane + metrics.time * 0.35) * (10 + metrics.mid * 70);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = `hsla(${210 + lane * 9 + metrics.noteIndex * 12}, 82%, 58%, ${0.16 + metrics.rms * 0.18})`;
      ctx.lineWidth = 1.1;
      ctx.stroke();
    }
    const cx = w / 2 + Math.sin(metrics.time * 0.9) * metrics.mid * w * 0.2;
    const cy = h / 2 + Math.cos(metrics.time * 0.7) * metrics.bass * h * 0.18;
    ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < 220; i++) {
      const a = i * 2.399 + metrics.time * (0.12 + metrics.treble * 0.4);
      const r = Math.sqrt(i) * (7 + metrics.rms * 20);
      const x = cx + Math.cos(a) * r;
      const y = cy + Math.sin(a) * r;
      ctx.fillStyle = `hsla(${metrics.noteIndex * 24 + i * 0.8}, 92%, ${38 + metrics.mid * 38}%, ${0.08 + i / 900})`;
      ctx.beginPath();
      ctx.arc(x, y, 1.2 + metrics.bass * 8 + (i % 9 === 0 ? metrics.rms * 10 : 0), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalCompositeOperation = "source-over";
    drawParticles(ctx, state, dt, "dot");
    vignette(ctx, w, h, 0.62);
    return;
  }

  if (mode === "ink") {
    drawBackground(ctx, w, h, "#050303");
    if (metrics.onset) spawn(state, metrics, w, h, 18);
    ctx.globalCompositeOperation = "lighter";
    const chordNotes = metrics.notes.length ? metrics.notes : [{ note: metrics.note, index: metrics.noteIndex, frequency: metrics.pitch ?? 220, strength: metrics.rms }];
    for (let n = 0; n < chordNotes.length; n++) {
      const note = chordNotes[n];
      const cx = w * (0.2 + ((note.index * 0.071 + n * 0.17) % 0.65));
      const cy = h * (0.2 + ((note.index * 0.113 + n * 0.23) % 0.6));
      for (let strand = 0; strand < 22; strand++) {
        ctx.beginPath();
        for (let i = 0; i < 120; i++) {
          const t = i / 119;
          const a = t * Math.PI * 2 * (1.5 + note.index * 0.08) + metrics.time * (0.35 + note.strength);
          const r = Math.pow(t, 0.55) * Math.min(w, h) * (0.12 + note.strength * 0.24);
          const noise = Math.sin(i * 0.37 + strand + metrics.time * 2) * (12 + metrics.treble * 80);
          const x = cx + Math.cos(a + strand) * r + noise;
          const y = cy + Math.sin(a * 1.7 + strand * 0.3) * r - noise * 0.3;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        glowLine(ctx, `hsla(${note.index * 31 + strand * 9}, 90%, ${45 + note.strength * 30}%, ${0.045 + note.strength * 0.13})`, 18, 0.8 + note.strength * 4);
        ctx.stroke();
      }
    }
    ctx.shadowBlur = 0;
    drawParticles(ctx, state, dt, "ink");
    ctx.globalCompositeOperation = "source-over";
    vignette(ctx, w, h, 0.74);
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
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true },
      });
      const audioCtx = new AudioContext();
      const source = audioCtx.createMediaStreamSource(stream);
      const inputGain = audioCtx.createGain();
      inputGain.gain.value = 2.8;
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      analyser.minDecibels = -96;
      analyser.maxDecibels = -12;
      analyser.smoothingTimeConstant = 0.58;
      source.connect(inputGain);
      inputGain.connect(analyser);
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
      const demoChords = [
        [220, 277.18, 329.63],
        [196, 246.94, 293.66, 369.99],
        [261.63, 329.63, 392],
        [146.83, 220, 293.66, 349.23],
      ];
      const fakeChord = demoChords[Math.floor(phase * 1.2) % demoChords.length];
      const fakePitch = fakeChord[0];
      const note = noteFromPitch(fakePitch);
      const notes = fakeChord.map((frequency, i) => ({
        ...notePartsFromFrequency(frequency),
        frequency,
        strength: 0.92 - i * 0.1,
      }));
      const rms = 0.18 + 0.14 * (Math.sin(phase * 3) * 0.5 + 0.5);
      const onset = Math.abs(Math.sin(phase * Math.PI)) > 0.94;
      return {
        waveform,
        freqData,
        metrics: {
          rms,
          bass: 0.48,
          mid: 0.38,
          treble: 0.32,
          centroid: 0.3,
          pitch: fakePitch,
          note: note.note,
          noteIndex: note.index,
          notes,
          chord: labelChord(notes, note.note),
          onset,
          time,
        },
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
    const spectrumNotes = detectSpectrumNotes(freqData, ctx.sampleRate, analyser.fftSize);
    const notes = spectrumNotes.length ? spectrumNotes : pitch ? [{ ...note, frequency: pitch, strength: liftedFallbackStrength(floatWave) }] : [];
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
    const liftedRms = clamp(Math.pow(rms * 7.5, 0.72));
    const onset = rms > 0.008 && rms > lastRmsRef.current * 1.28;
    lastRmsRef.current = lerp(lastRmsRef.current, rms, 0.34);
    return {
      waveform,
      freqData,
      metrics: {
        rms: liftedRms,
        bass: clamp(Math.pow(bass * 1.35, 0.8)),
        mid: clamp(Math.pow(mid * 1.45, 0.82)),
        treble: clamp(Math.pow(treble * 1.6, 0.86)),
        centroid,
        pitch,
        note: note.note,
        noteIndex: note.index,
        notes,
        chord: labelChord(notes, note.note),
        onset,
        time,
      },
    };
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
          <span>{metrics.chord}</span>
          <small>{metrics.notes.length ? metrics.notes.map((n) => n.note).join(" · ") : `${Math.round(metrics.rms * 100)}%`}</small>
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
