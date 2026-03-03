import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'
import { MicVAD } from '@ricky0123/vad-web'

const WASM_PATH  = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
const MODEL_URL  = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
const EMA_ALPHA  = 0.2

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
]

// ── DOM ──────────────────────────────────────────────────────────────────────

const gameCanvas   = document.getElementById('game')
const videoEl      = document.getElementById('video')
const overlayEl    = document.getElementById('overlay')
const statusEl      = document.getElementById('status')
const voiceStatusEl = document.getElementById('voice-status')
const btnStart      = document.getElementById('btnStart')
const startOverlay  = document.getElementById('startOverlay')

let handLandmarker    = null
let animFrameId       = null
let smoothedAngle     = 0
let lastDetectionTime = 0
let pendingSpeedTarget = null
let workerBusy        = false
let musicCtx          = null
let musicLoopId       = null
const DETECTION_INTERVAL_MS = 33   // ~30 fps for hand detection

function setStatus(msg) { statusEl.textContent = msg }
function setVoiceStatus(msg) { if (voiceStatusEl) voiceStatusEl.textContent = msg }

// ── Canvas constants ──────────────────────────────────────────────────────────

const GW = 700
const GH = 525
const HORIZON_Y    = GH * 0.40   // sky / ground split
const ROAD_BOTTOM  = GH * 0.725  // road meets dashboard
const ROAD_HW_TOP  = GW * 0.060  // road half-width at horizon
const ROAD_HW_BTM  = GW * 0.425  // road half-width at near edge

// ── Perspective helpers ───────────────────────────────────────────────────────

// t: 0 = at horizon, 1 = at near edge (ROAD_BOTTOM)
function syAt(t) { return HORIZON_Y + (ROAD_BOTTOM - HORIZON_Y) * (t * t) }
function hwAt(t) { return ROAD_HW_TOP + (ROAD_HW_BTM - ROAD_HW_TOP) * t }
function cxAt(t, vpX) { return vpX + (GW / 2 - vpX) * t }

// ── Scenery objects ───────────────────────────────────────────────────────────

function makeObject(i) {
  const type = Math.random() < 0.62 ? 'tree' : 'building'
  const base = {
    side:    i % 2 === 0 ? 'left' : 'right',
    t:       Math.random(),        // depth: 0=far, 1=near
    lateral: 0.04 + Math.random() * 0.09,
    hue:     Math.random() * 40 | 0,
    type,
  }
  if (type === 'building') {
    const rows = 2 + (Math.random() * 3 | 0)
    const cols = 2
    return {
      ...base,
      bW:      18 + (Math.random() * 22 | 0),
      bH:      50 + (Math.random() * 85 | 0),
      winRows: rows,
      winCols: cols,
      windows: Array.from({ length: rows * cols }, () => Math.random() > 0.35),
    }
  }
  return base
}

const N_OBJECTS = 30
const gameState = {
  roadOffset: 0,
  speedKmh:   100,
  speed:      0.010,   // derived; always = speedKmh * 0.0001
  lightsOn:   false,
  musicOn:    false,
  objects:    Array.from({ length: N_OBJECTS }, (_, i) => makeObject(i)),
}

function respawnObject(obj, i) {
  const fresh = makeObject(i)
  Object.assign(obj, fresh)
  obj.t    = Math.random() * 0.12
  obj.side = Math.random() < 0.5 ? 'left' : 'right'
}

// ── City skyline (static silhouette at horizon) ───────────────────────────────

// [centerX fraction, height fraction of HORIZON_Y, width fraction of GW]
const SKYLINE = [
  [0.04,  0.80, 0.06], [0.10,  0.55, 0.045], [0.16,  0.92, 0.07],
  [0.23,  0.62, 0.05], [0.29,  0.88, 0.075], [0.36,  0.50, 0.04],
  [0.42,  0.78, 0.06], [0.49,  0.45, 0.05],  [0.55,  0.85, 0.065],
  [0.62,  0.68, 0.055],[0.69,  0.92, 0.07],  [0.76,  0.58, 0.05],
  [0.83,  0.82, 0.065],[0.90,  0.70, 0.055], [0.96,  0.90, 0.06],
]

function drawSkyline(ctx, steerFrac) {
  const shift = steerFrac * GW * 0.08   // parallax

  // Building silhouettes
  ctx.fillStyle = '#080c1a'
  for (const [xf, hf, wf] of SKYLINE) {
    const bH = HORIZON_Y * hf
    const bW = wf * GW
    const bX = xf * GW + shift - GW * 0.04 - bW / 2
    ctx.fillRect(bX, HORIZON_Y - bH, bW, bH)
  }

  // Lit windows (deterministic pattern)
  for (const [xf, hf, wf] of SKYLINE) {
    const bH  = HORIZON_Y * hf
    const bW  = wf * GW
    const bX  = xf * GW + shift - GW * 0.04 - bW / 2
    const rows = Math.max(2, bH / 13 | 0)
    const cols = Math.max(1, bW / 8  | 0)
    const seed = (xf * 1000) | 0
    ctx.fillStyle = 'rgba(255,215,110,0.38)'
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (((r * 7 + c * 3 + seed) % 3) !== 0) {
          ctx.fillRect(
            bX + (c + 0.15) * bW / cols,
            HORIZON_Y - bH + (r + 0.18) * bH / rows,
            bW / cols * 0.60,
            bH / rows * 0.55,
          )
        }
      }
    }
  }
}

// ── Scenery object drawing ────────────────────────────────────────────────────

function drawSceneryObject(ctx, obj, vpX) {
  const { t, side, lateral, hue, type } = obj
  if (t < 0.06 || t > 0.97) return

  const sy  = syAt(t)
  if (sy > ROAD_BOTTOM - 2) return

  const cx  = cxAt(t, vpX)
  const hw  = hwAt(t)
  const margin = lateral * GW * t
  const bx  = side === 'left' ? cx - hw - margin : cx + hw + margin

  if (type === 'tree') {
    const trunkH = 52 * t
    const trunkW = Math.max(1.5, 7 * t)
    const lr     = 24 * t   // leaf radius

    // Trunk
    ctx.fillStyle = `hsl(25,52%,${13 + 14 * t | 0}%)`
    ctx.fillRect(bx - trunkW / 2, sy - trunkH, trunkW, trunkH)

    // Foliage layers (back-to-front shading)
    ctx.beginPath(); ctx.arc(bx, sy - trunkH, lr * 1.1, 0, Math.PI * 2)
    ctx.fillStyle = `hsl(${112 + hue},52%,${9 + 11 * t | 0}%)`; ctx.fill()

    ctx.beginPath(); ctx.arc(bx, sy - trunkH, lr * 0.9, 0, Math.PI * 2)
    ctx.fillStyle = `hsl(${117 + hue},58%,${16 + 17 * t | 0}%)`; ctx.fill()

    ctx.beginPath(); ctx.arc(bx - lr * 0.22, sy - trunkH - lr * 0.32, lr * 0.58, 0, Math.PI * 2)
    ctx.fillStyle = `hsl(${121 + hue},62%,${22 + 20 * t | 0}%)`; ctx.fill()

  } else {
    const bW = obj.bW * t
    const bH = obj.bH * t
    if (bH < 2 || bW < 1) return

    const rx = bx - bW / 2
    const ry = sy - bH

    // Main facade
    ctx.fillStyle = `hsl(220,11%,${10 + 17 * t | 0}%)`
    ctx.fillRect(rx, ry, bW, bH)

    // Top band
    ctx.fillStyle = `hsl(220,9%,${7 + 10 * t | 0}%)`
    ctx.fillRect(rx + bW * 0.06, ry, bW * 0.88, bH * 0.09)

    // Windows
    const { winRows: rows, winCols: cols, windows } = obj
    ctx.fillStyle = `rgba(255,215,95,${0.18 + 0.48 * t})`
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (windows[r * cols + c]) {
          ctx.fillRect(
            rx + (c + 0.14) * bW / cols,
            ry + (r + 0.18) * bH / rows,
            bW / cols * 0.65,
            bH / rows * 0.52,
          )
        }
      }
    }
  }
}

// ── Car interior ──────────────────────────────────────────────────────────────

function drawCarInterior(ctx, steerAngle) {
  const W = GW, H = GH
  const steerFrac = steerAngle / 90

  // A-pillars (left and right windshield columns)
  const pillarW = 28
  ctx.save()

  // Left A-pillar
  ctx.beginPath()
  ctx.moveTo(0, 0)
  ctx.lineTo(pillarW * 2, 0)
  ctx.lineTo(pillarW, ROAD_BOTTOM)
  ctx.lineTo(0, H)
  ctx.fillStyle = '#0e0e0e'
  ctx.fill()

  // Right A-pillar
  ctx.beginPath()
  ctx.moveTo(W, 0)
  ctx.lineTo(W - pillarW * 2, 0)
  ctx.lineTo(W - pillarW, ROAD_BOTTOM)
  ctx.lineTo(W, H)
  ctx.fillStyle = '#0e0e0e'
  ctx.fill()

  // Hood visible at the base of the windshield
  const hoodY  = ROAD_BOTTOM - 10
  const leanX  = steerFrac * W * 0.018
  const hoodGrad = ctx.createLinearGradient(0, hoodY, 0, ROAD_BOTTOM)
  hoodGrad.addColorStop(0, '#181818')
  hoodGrad.addColorStop(1, '#0c0c0c')

  ctx.beginPath()
  ctx.moveTo(W * 0.32 + leanX, hoodY)
  ctx.lineTo(W * 0.68 + leanX, hoodY)
  ctx.bezierCurveTo(W * 0.82, ROAD_BOTTOM - 2, W * 0.92, ROAD_BOTTOM, W, ROAD_BOTTOM)
  ctx.lineTo(0, ROAD_BOTTOM)
  ctx.bezierCurveTo(W * 0.08, ROAD_BOTTOM, W * 0.18, ROAD_BOTTOM - 2, W * 0.32 + leanX, hoodY)
  ctx.closePath()
  ctx.fillStyle = hoodGrad
  ctx.fill()

  // Hood center ridge
  ctx.beginPath()
  ctx.moveTo(W / 2 + leanX * 0.4, hoodY + 2)
  ctx.lineTo(W / 2, ROAD_BOTTOM)
  ctx.strokeStyle = 'rgba(55,55,55,0.5)'
  ctx.lineWidth = 1
  ctx.stroke()

  ctx.restore()

  // Dashboard body
  const dashY = ROAD_BOTTOM
  const dashH = H - dashY

  const dashPath = () => {
    ctx.beginPath()
    ctx.moveTo(0, dashY + dashH * 0.38)
    ctx.bezierCurveTo(W * 0.07, dashY - dashH * 0.02, W * 0.35, dashY - dashH * 0.06, W / 2, dashY - dashH * 0.06)
    ctx.bezierCurveTo(W * 0.65, dashY - dashH * 0.06, W * 0.93, dashY - dashH * 0.02, W, dashY + dashH * 0.38)
    ctx.lineTo(W, H)
    ctx.lineTo(0, H)
    ctx.closePath()
  }

  // Shadow beneath dashboard curve
  ctx.save()
  dashPath()
  const shadowGrad = ctx.createLinearGradient(0, dashY - 8, 0, H)
  shadowGrad.addColorStop(0, '#141414')
  shadowGrad.addColorStop(0.35, '#1c1c1c')
  shadowGrad.addColorStop(1, '#090909')
  ctx.fillStyle = shadowGrad
  ctx.fill()
  ctx.restore()

  // Blue LED accent strip along dashboard top edge
  ctx.save()
  dashPath()
  ctx.clip()
  ctx.beginPath()
  ctx.moveTo(W * 0.08, dashY + dashH * 0.34)
  ctx.bezierCurveTo(W * 0.28, dashY + dashH * 0.04, W * 0.42, dashY - dashH * 0.02, W / 2, dashY - dashH * 0.02)
  ctx.bezierCurveTo(W * 0.58, dashY - dashH * 0.02, W * 0.72, dashY + dashH * 0.04, W * 0.92, dashY + dashH * 0.34)
  ctx.strokeStyle = 'rgba(0,160,255,0.22)'
  ctx.lineWidth = 2.5
  ctx.stroke()
  ctx.restore()

  // Steering wheel
  drawSteeringWheel(ctx, W, H, dashY, dashH, steerAngle)
}

function drawSteeringWheel(ctx, W, H, dashY, dashH, steerAngle) {
  const cx = W / 2
  const cy = dashY + dashH * 0.44
  const R  = dashH * 0.50

  ctx.save()
  ctx.translate(cx, cy)
  ctx.rotate(-steerAngle * Math.PI / 180)

  // Drop shadow
  ctx.shadowColor = 'rgba(0,0,0,0.9)'
  ctx.shadowBlur  = 18

  // Outer rim (dark base)
  ctx.beginPath()
  ctx.arc(0, 0, R, 0, Math.PI * 2)
  ctx.strokeStyle = '#141414'
  ctx.lineWidth   = R * 0.22
  ctx.stroke()

  // Rim surface
  ctx.beginPath()
  ctx.arc(0, 0, R, 0, Math.PI * 2)
  ctx.strokeStyle = '#2c2c2c'
  ctx.lineWidth   = R * 0.17
  ctx.stroke()

  // Rim highlight (thin lighter arc, top)
  ctx.beginPath()
  ctx.arc(0, 0, R - R * 0.05, -Math.PI * 0.75, -Math.PI * 0.25)
  ctx.strokeStyle = 'rgba(85,85,85,0.65)'
  ctx.lineWidth   = 1.5
  ctx.stroke()

  ctx.shadowBlur = 0

  // Three spokes at 90°, 210°, 330°
  ctx.lineCap = 'round'
  for (const aDeg of [90, 210, 330]) {
    const a  = aDeg * Math.PI / 180
    const sx = Math.cos(a)
    const sy = Math.sin(a)
    ctx.beginPath()
    ctx.moveTo(sx * R * 0.14, sy * R * 0.14)
    ctx.lineTo(sx * R * 0.82, sy * R * 0.82)
    ctx.strokeStyle = '#252525'
    ctx.lineWidth   = R * 0.13
    ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(sx * R * 0.14, sy * R * 0.14)
    ctx.lineTo(sx * R * 0.82, sy * R * 0.82)
    ctx.strokeStyle = 'rgba(65,65,65,0.7)'
    ctx.lineWidth   = R * 0.06
    ctx.stroke()
  }
  ctx.lineCap = 'butt'

  // Hub
  ctx.beginPath(); ctx.arc(0, 0, R * 0.17, 0, Math.PI * 2)
  ctx.fillStyle = '#222'; ctx.fill()
  ctx.beginPath(); ctx.arc(0, 0, R * 0.12, 0, Math.PI * 2)
  ctx.fillStyle = '#333'; ctx.fill()
  ctx.beginPath(); ctx.arc(0, 0, R * 0.042, 0, Math.PI * 2)
  ctx.fillStyle = 'rgba(0,180,255,0.85)'; ctx.fill()

  ctx.restore()
}

// ── Road markings ─────────────────────────────────────────────────────────────

function drawRoad(ctx, vpX) {
  // Ground (grass outside road)
  const grassGrad = ctx.createLinearGradient(0, HORIZON_Y, 0, ROAD_BOTTOM)
  grassGrad.addColorStop(0, '#111f08')
  grassGrad.addColorStop(1, '#1c3a0c')
  ctx.fillStyle = grassGrad
  ctx.fillRect(0, HORIZON_Y, GW, ROAD_BOTTOM - HORIZON_Y)

  // Road trapezoid
  ctx.beginPath()
  ctx.moveTo(vpX - ROAD_HW_TOP, HORIZON_Y)
  ctx.lineTo(vpX + ROAD_HW_TOP, HORIZON_Y)
  ctx.lineTo(GW / 2 + ROAD_HW_BTM, ROAD_BOTTOM)
  ctx.lineTo(GW / 2 - ROAD_HW_BTM, ROAD_BOTTOM)
  ctx.closePath()
  const roadGrad = ctx.createLinearGradient(0, HORIZON_Y, 0, ROAD_BOTTOM)
  roadGrad.addColorStop(0, '#1c1c1c')
  roadGrad.addColorStop(0.45, '#2e2e2e')
  roadGrad.addColorStop(1, '#444')
  ctx.fillStyle = roadGrad
  ctx.fill()

  // Road shoulder strips
  for (const sign of [-1, 1]) {
    ctx.beginPath()
    ctx.moveTo(vpX + sign * ROAD_HW_TOP, HORIZON_Y)
    ctx.lineTo(vpX + sign * (ROAD_HW_TOP + GW * 0.038), HORIZON_Y)
    ctx.lineTo(GW / 2 + sign * (ROAD_HW_BTM + GW * 0.038), ROAD_BOTTOM)
    ctx.lineTo(GW / 2 + sign * ROAD_HW_BTM, ROAD_BOTTOM)
    ctx.closePath()
    ctx.fillStyle = '#222'
    ctx.fill()
  }

  // Lane markings
  const N = 15
  for (let i = 0; i < N; i++) {
    const t0 = ((i / N + gameState.roadOffset) % 1.0)
    if (t0 < 0.03) continue

    const t1  = Math.min(1.0, t0 + 0.38 / N)
    const y0  = syAt(t0)
    const y1  = Math.min(syAt(t1), ROAD_BOTTOM)
    if (y1 <= y0) continue

    const cx   = cxAt(t0, vpX)
    const hw   = hwAt(t0)
    const segH = y1 - y0

    // Center dash (yellow)
    const dw = Math.max(1.5, hw * 0.021)
    ctx.fillStyle = 'rgba(255,225,70,0.88)'
    ctx.fillRect(cx - dw, y0, dw * 2, segH)

    // Edge lines (white)
    const ew = Math.max(1, hw * 0.028)
    ctx.fillStyle = 'rgba(255,255,255,0.72)'
    ctx.fillRect(cx - hw,      y0, ew, segH)
    ctx.fillRect(cx + hw - ew, y0, ew, segH)
  }
}

// ── Music synthesizer (techno / electro) ─────────────────────────────────────
const TECHNO_BPM = 132
const STEP       = 60 / TECHNO_BPM / 4   // 16th-note step in seconds

// 16-step drum + bass patterns (one bar loops forever)
const KICK_PAT  = [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0]   // 4-on-the-floor
const HIHAT_PAT = [0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,1]   // 8th-notes + roll
const SNARE_PAT = [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0]   // 2 and 4
// Bass: MIDI notes (C1=24), 0 = rest — classic driving techno line
const BASS_PAT  = [36,0,36,39, 0,36,0,43, 41,0,39,0, 36,0,43,41]

function midiToHz(midi) { return 440 * Math.pow(2, (midi - 69) / 12) }

function scheduleKick(ctx, t) {
  const osc = ctx.createOscillator()
  const g   = ctx.createGain()
  osc.frequency.setValueAtTime(160, t)
  osc.frequency.exponentialRampToValueAtTime(40, t + 0.08)
  g.gain.setValueAtTime(1.2, t)
  g.gain.exponentialRampToValueAtTime(0.001, t + 0.28)
  osc.connect(g); g.connect(ctx.destination)
  osc.start(t); osc.stop(t + 0.32)
}

function scheduleHihat(ctx, t, open) {
  const dur = open ? 0.10 : 0.035
  const buf  = ctx.createBuffer(1, ctx.sampleRate * dur, ctx.sampleRate)
  const data = buf.getChannelData(0)
  for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1
  const src    = ctx.createBufferSource()
  const filter = ctx.createBiquadFilter()
  const g      = ctx.createGain()
  src.buffer         = buf
  filter.type        = 'highpass'
  filter.frequency.value = 7500
  g.gain.setValueAtTime(0.25, t)
  g.gain.exponentialRampToValueAtTime(0.001, t + dur)
  src.connect(filter); filter.connect(g); g.connect(ctx.destination)
  src.start(t)
}

function scheduleSnare(ctx, t) {
  // Noise body
  const dur  = 0.14
  const buf  = ctx.createBuffer(1, ctx.sampleRate * dur, ctx.sampleRate)
  const data = buf.getChannelData(0)
  for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1
  const src    = ctx.createBufferSource()
  const filter = ctx.createBiquadFilter()
  const g      = ctx.createGain()
  src.buffer         = buf
  filter.type        = 'bandpass'
  filter.frequency.value = 2200
  filter.Q.value     = 0.8
  g.gain.setValueAtTime(0.7, t)
  g.gain.exponentialRampToValueAtTime(0.001, t + dur)
  src.connect(filter); filter.connect(g); g.connect(ctx.destination)
  src.start(t)
  // Tone click
  const osc = ctx.createOscillator()
  const og  = ctx.createGain()
  osc.frequency.value = 220
  og.gain.setValueAtTime(0.5, t)
  og.gain.exponentialRampToValueAtTime(0.001, t + 0.06)
  osc.connect(og); og.connect(ctx.destination)
  osc.start(t); osc.stop(t + 0.07)
}

function scheduleBass(ctx, t, midi) {
  const osc    = ctx.createOscillator()
  const filter = ctx.createBiquadFilter()
  const g      = ctx.createGain()
  osc.type           = 'sawtooth'
  osc.frequency.value = midiToHz(midi)
  filter.type        = 'lowpass'
  filter.frequency.setValueAtTime(900, t)
  filter.frequency.exponentialRampToValueAtTime(280, t + STEP * 0.88)
  filter.Q.value     = 4
  g.gain.setValueAtTime(0.55, t)
  g.gain.exponentialRampToValueAtTime(0.001, t + STEP * 0.82)
  osc.connect(filter); filter.connect(g); g.connect(ctx.destination)
  osc.start(t); osc.stop(t + STEP)
}

function scheduleBar(ctx, startTime) {
  for (let i = 0; i < 16; i++) {
    const t = startTime + i * STEP
    if (KICK_PAT[i])  scheduleKick(ctx, t)
    if (HIHAT_PAT[i]) scheduleHihat(ctx, t, i === 14)   // open hat on last step
    if (SNARE_PAT[i]) scheduleSnare(ctx, t)
    if (BASS_PAT[i])  scheduleBass(ctx, t, BASS_PAT[i])
  }
  return startTime + 16 * STEP
}

function startMusic() {
  if (musicCtx) return
  gameState.musicOn = true
  musicCtx = new AudioContext()
  let nextStart = musicCtx.currentTime + 0.05
  function loop() {
    nextStart = scheduleBar(musicCtx, nextStart)
    const delay = (nextStart - musicCtx.currentTime - 0.3) * 1000
    musicLoopId = setTimeout(loop, Math.max(0, delay))
  }
  loop()
}

function stopMusic() {
  if (!musicCtx) return
  gameState.musicOn = false
  clearTimeout(musicLoopId)
  musicLoopId = null
  musicCtx.close()
  musicCtx = null
}

// ── Headlights ────────────────────────────────────────────────────────────────

function drawHeadlights(ctx, vpX) {
  if (!gameState.lightsOn) return
  ctx.save()

  // Clip to road trapezoid so light stays on the road surface
  ctx.beginPath()
  ctx.moveTo(vpX - ROAD_HW_TOP, HORIZON_Y)
  ctx.lineTo(vpX + ROAD_HW_TOP, HORIZON_Y)
  ctx.lineTo(GW / 2 + ROAD_HW_BTM, ROAD_BOTTOM)
  ctx.lineTo(GW / 2 - ROAD_HW_BTM, ROAD_BOTTOM)
  ctx.closePath()
  ctx.clip()

  // Two beams, one per headlight, each covering half the road width
  for (const sign of [-1, 1]) {
    // Radial gradient originating from headlight position just below ROAD_BOTTOM
    const hx   = GW / 2 + sign * GW * 0.085
    const hy   = ROAD_BOTTOM + 60
    const grad = ctx.createRadialGradient(hx, hy, 15, hx, hy, GH * 0.72)
    grad.addColorStop(0,    'rgba(255,252,195,0.28)')
    grad.addColorStop(0.30, 'rgba(255,250,188,0.16)')
    grad.addColorStop(0.65, 'rgba(255,248,180,0.06)')
    grad.addColorStop(1,    'rgba(255,245,170,0)')

    // Trapezoid: near edge spans center → outer, far edge narrows to vanishing point
    ctx.beginPath()
    ctx.moveTo(GW / 2 + sign * GW * 0.002, ROAD_BOTTOM)
    ctx.lineTo(GW / 2 + sign * ROAD_HW_BTM, ROAD_BOTTOM)
    ctx.lineTo(vpX + sign * ROAD_HW_TOP, HORIZON_Y)
    ctx.lineTo(vpX, HORIZON_Y)
    ctx.closePath()
    ctx.fillStyle = grad
    ctx.fill()
  }

  ctx.restore()
}

// ── HUD ──────────────────────────────────────────────────────────────────────

function drawHUD(ctx, steerAngle) {
  const speedKmh = gameState.speedKmh
  ctx.save()
  ctx.textAlign    = 'right'
  ctx.textBaseline = 'top'
  ctx.font         = `bold ${GH * 0.072 | 0}px monospace`
  ctx.fillStyle    = '#00ff99'
  ctx.fillText(`${speedKmh}`, GW - 14, 14)
  ctx.font         = `${GH * 0.024 | 0}px monospace`
  ctx.fillStyle    = 'rgba(0,230,140,0.65)'
  ctx.fillText('km/h', GW - 14, 14 + GH * 0.075)

  // Lights indicator
  ctx.font      = `${GH * 0.022 | 0}px monospace`
  ctx.fillStyle = gameState.lightsOn ? 'rgba(255,252,120,0.92)' : 'rgba(255,255,255,0.22)'
  ctx.fillText('◉ LIGHTS', GW - 14, 14 + GH * 0.115)

  // Music indicator
  ctx.fillStyle = gameState.musicOn ? 'rgba(100,200,255,0.92)' : 'rgba(255,255,255,0.22)'
  ctx.fillText('♪ MUSIC', GW - 14, 14 + GH * 0.155)
  ctx.restore()
}

// ── Full scene draw ───────────────────────────────────────────────────────────

function drawGame(steerAngle) {
  const ctx = gameCanvas.getContext('2d')
  const steerFrac = Math.max(-1, Math.min(1, steerAngle / 90))
  const vpX = GW / 2 - steerFrac * GW * 0.18

  // 1. Sky gradient
  const skyGrad = ctx.createLinearGradient(0, 0, 0, HORIZON_Y)
  skyGrad.addColorStop(0,   '#04071a')
  skyGrad.addColorStop(0.5, '#0c1e52')
  skyGrad.addColorStop(1,   '#183a7a')
  ctx.fillStyle = skyGrad
  ctx.fillRect(0, 0, GW, HORIZON_Y + 1)

  // 2. City skyline silhouette
  drawSkyline(ctx, steerFrac)

  // 3. Horizon glow (orange sunset strip)
  const glowGrad = ctx.createLinearGradient(0, HORIZON_Y - 32, 0, HORIZON_Y + 14)
  glowGrad.addColorStop(0,    'rgba(255,130,20,0)')
  glowGrad.addColorStop(0.65, 'rgba(255,90,10,0.28)')
  glowGrad.addColorStop(1,    'rgba(200,50,0,0.12)')
  ctx.fillStyle = glowGrad
  ctx.fillRect(0, HORIZON_Y - 32, GW, 46)

  // 4. Road + grass + lane markings
  drawRoad(ctx, vpX)

  // 4b. Headlight beams (on top of road surface, under scenery)
  drawHeadlights(ctx, vpX)

  // 5. Scenery objects (sorted back-to-front)
  const sorted = gameState.objects.slice().sort((a, b) => a.t - b.t)
  for (const obj of sorted) drawSceneryObject(ctx, obj, vpX)

  // 6. Car interior (hood, A-pillars, dashboard, steering wheel)
  drawCarInterior(ctx, steerAngle)

  // 7. Speed HUD
  drawHUD(ctx, steerAngle)
}

// ── Hand skeleton overlay (for PiP webcam) ────────────────────────────────────

function drawHandSkeleton(ctx, landmarks, W, H) {
  ctx.strokeStyle = 'rgba(255,255,255,0.38)'
  ctx.lineWidth   = 1
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath()
    ctx.moveTo(landmarks[a].x * W, landmarks[a].y * H)
    ctx.lineTo(landmarks[b].x * W, landmarks[b].y * H)
    ctx.stroke()
  }
  ctx.beginPath()
  ctx.arc(landmarks[0].x * W, landmarks[0].y * H, 4, 0, Math.PI * 2)
  ctx.fillStyle = '#ff6'
  ctx.fill()
}

// ── Main loop ─────────────────────────────────────────────────────────────────

function loop() {
  animFrameId = requestAnimationFrame(loop)

  if (pendingSpeedTarget !== null) {
    gameState.speedKmh = pendingSpeedTarget
    gameState.speed    = gameState.speedKmh * 0.0001
    pendingSpeedTarget = null
  }

  // Advance road scroll
  gameState.roadOffset = (gameState.roadOffset + gameState.speed) % 1.0

  // Advance scenery objects toward viewer
  for (let i = 0; i < gameState.objects.length; i++) {
    gameState.objects[i].t += gameState.speed * 0.88
    if (gameState.objects[i].t > 1.0) respawnObject(gameState.objects[i], i)
  }

  // Hand detection — throttled so it doesn't stall the render loop
  const now = performance.now()
  if (now - lastDetectionTime >= DETECTION_INTERVAL_MS) {
    lastDetectionTime = now

  const W   = overlayEl.width
  const H   = overlayEl.height
  const ctx = overlayEl.getContext('2d')
  ctx.clearRect(0, 0, W, H)

  const results = handLandmarker.detectForVideo(videoEl, now)

  if (results.landmarks.length === 2) {
    for (const hand of results.landmarks) drawHandSkeleton(ctx, hand, W, H)

    const [left, right] = results.landmarks.slice().sort((a, b) => a[0].x - b[0].x)
    const lw = left[0], rw = right[0]

    ctx.beginPath()
    ctx.moveTo(lw.x * W, lw.y * H)
    ctx.lineTo(rw.x * W, rw.y * H)
    ctx.strokeStyle = '#4af'
    ctx.lineWidth   = 1.5
    ctx.stroke()

    const raw     = Math.atan2(rw.y - lw.y, rw.x - lw.x) * 180 / Math.PI
    const clamped = Math.max(-90, Math.min(90, raw))
    smoothedAngle = EMA_ALPHA * clamped + (1 - EMA_ALPHA) * smoothedAngle
    setStatus('Hands detected — steering active')
  } else {
    smoothedAngle *= (1 - EMA_ALPHA)
    setStatus(results.landmarks.length === 0 ? 'Show both hands to steer' : 'Show both hands')
  }

  } // end detection throttle

  drawGame(smoothedAngle)
}

// ── Start / Stop ──────────────────────────────────────────────────────────────

async function start() {
  btnStart.disabled = true
  setStatus('Loading MediaPipe...')

  if (!handLandmarker) {
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH)
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numHands:    2,
    })
  }

  setStatus('Starting webcam...')
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: 'user' },
  })
  videoEl.srcObject = stream
  await new Promise(resolve => { videoEl.onloadedmetadata = resolve })
  videoEl.play()

  overlayEl.width  = videoEl.videoWidth
  overlayEl.height = videoEl.videoHeight

  startOverlay.style.display = 'none'

  loop()

  // Launch audio worker
  const audioWorker = new Worker(new URL('./audio-worker.js', import.meta.url), { type: 'module' })
  audioWorker.onmessage = ({ data }) => {
    if (data.type === 'ready') {
      setVoiceStatus('Voice ready — speed / slow / lights on / lights off / music / stop music')
      startVAD(audioWorker)
    }
    if (data.type === 'progress') {
      setVoiceStatus(`Loading audio model… ${(data.progress.progress * 100 | 0)}%`)
    }
    if (data.type === 'result') {
      workerBusy = false
      if (data.keyword === 'speed')           pendingSpeedTarget = 120
      else if (data.keyword === 'slow')       pendingSpeedTarget = 0
      else if (data.keyword === 'lights_on')  gameState.lightsOn = true
      else if (data.keyword === 'lights_off') gameState.lightsOn = false
      else if (data.keyword === 'music_on')   startMusic()
      else if (data.keyword === 'music_off')  stopMusic()
    }
  }
  audioWorker.postMessage({ type: 'load' })
}

async function startVAD(worker) {
  const vad = await MicVAD.new({
    baseAssetPath: '/',   // worklet + model both load from /public
    model: 'v5',          // silero_vad_v5.onnx
    ortConfig(ort) {
      // Same CDN fix as the audio worker — vad-web shares onnxruntime-web 1.24.2
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/'
    },
    onSpeechEnd(audio) {
      // audio is Float32Array @ 16 kHz — already decoded, no MediaRecorder needed
      if (workerBusy) return
      workerBusy = true
      const copy = new Float32Array(audio)   // copy so we can safely transfer the buffer
      worker.postMessage(
        { type: 'transcribe', audioData: copy.buffer, sampleRate: 16000 },
        [copy.buffer]                        // zero-copy transfer
      )
    },
  })
  vad.start()
}

function stop() {
  if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null }
  videoEl.srcObject?.getTracks().forEach(t => t.stop())
  videoEl.srcObject = null
  overlayEl.getContext('2d').clearRect(0, 0, overlayEl.width, overlayEl.height)
  smoothedAngle = 0
  stopMusic()
  startOverlay.style.display = 'flex'
  setStatus('Click Start to begin.')
  btnStart.textContent = 'Start'
  btnStart.disabled    = false
  btnStart.onclick     = start
}

// ── Init ──────────────────────────────────────────────────────────────────────

btnStart.onclick = start
drawGame(0)
