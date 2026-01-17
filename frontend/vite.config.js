import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    host: true,
    allowedHosts: [
      '4aea88e3.r11.vip.cpolar.cn',
      'localhost',
      '127.0.0.1'
    ]
  }
})