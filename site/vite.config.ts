import { defineConfig } from "vite";

// Relative base so the built site works under any GitHub Pages sub-path and
// from a plain static server. No external hosts, no CDN, no runtime network.
export default defineConfig({
  base: "./",
  build: {
    target: "es2022",
    outDir: "dist",
    assetsInlineLimit: 0,
    cssCodeSplit: false,
  },
  server: { host: "127.0.0.1", port: 8874, strictPort: true },
  preview: { host: "127.0.0.1", port: 8874, strictPort: true },
});
