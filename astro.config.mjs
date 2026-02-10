import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import tailwind from "@astrojs/tailwind";
import sitemap from "@astrojs/sitemap";

export default defineConfig({
  site: "https://danielshort.me",
  output: "static",
  integrations: [
    react(),
    tailwind({
      applyBaseStyles: false
    }),
    sitemap()
  ],
  vite: {
    optimizeDeps: {
      include: ["onnxruntime-web"]
    }
  }
});
