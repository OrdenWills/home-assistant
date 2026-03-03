import * as ort from 'onnxruntime-web'
import AudioModel from './audio-model.js'

// Vite dev server doesn't serve node_modules .wasm files at the URL onnxruntime-web
// expects, so point it at the CDN instead.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/'

const MODEL_PATH = 'https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-ONNX/resolve/main'
// Each set covers the target word plus common ASR substitutions.
// "slow" is often transcribed as "stop" by the model (same initial consonant,
// vowel reduction), so both are included in SLOWER_WORDS.
const FASTER_WORDS     = ['speed', 'fast', 'accelerat', 'zoom', 'go', 'start']
const SLOWER_WORDS     = ['slow', 'stop', 'brake', 'deceler', 'easy']
const LIGHTS_ON_WORDS  = ['lights on', 'light on', 'headlight', 'brights', 'lines on']
const LIGHTS_OFF_WORDS = ['lights off', 'light off', 'lines off', 'lines of']
const MUSIC_ON_WORDS   = ['music', 'muse', 'play', 'song', 'tune']
const MUSIC_OFF_WORDS  = ['stop music', 'no music', 'silence', 'mute', 'quiet',
                          'music off', 'music of', 'music love']

let model = null

self.onmessage = async ({ data }) => {
  if (data.type === 'load') {
    model = new AudioModel()
    await model.load(MODEL_PATH, {
      device: navigator.gpu ? 'webgpu' : 'wasm',
      quantization: { decoder: 'q4', audioEncoder: 'q4',
                      audioEmbedding: 'q4', audioDetokenizer: 'q4', vocoder: 'q4' },
      progressCallback: p => self.postMessage({ type: 'progress', progress: p }),
    })
    self.postMessage({ type: 'ready' })
  }

  if (data.type === 'transcribe') {
    const { audioData, sampleRate } = data
    const text = await model.transcribe(new Float32Array(audioData), sampleRate)
    const lower = text.toLowerCase()
    let keyword = null
    if (LIGHTS_OFF_WORDS.some(w => lower.includes(w)))     keyword = 'lights_off'
    else if (LIGHTS_ON_WORDS.some(w => lower.includes(w)))  keyword = 'lights_on'
    else if (MUSIC_OFF_WORDS.some(w => lower.includes(w)))  keyword = 'music_off'
    else if (MUSIC_ON_WORDS.some(w => lower.includes(w)))   keyword = 'music_on'
    else if (FASTER_WORDS.some(w => lower.includes(w)))     keyword = 'speed'
    else if (SLOWER_WORDS.some(w => lower.includes(w)))     keyword = 'slow'
    self.postMessage({ type: 'result', text, keyword })
  }
}
