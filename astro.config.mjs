import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://danielshort.me',
  output: 'static',
  integrations: [react(), sitemap()],
  vite: {
    optimizeDeps: {
      include: ['onnxruntime-web']
    }
  }
});
