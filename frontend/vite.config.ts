// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // any request to /api/* will be forwarded to your json-server
      '/api/': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})
