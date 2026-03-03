import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    port: 3001,
  },
  optimizeDeps: {
    exclude: ['@mediapipe/tasks-vision'],
  },
  build: {
    target: 'esnext',
  },
})
