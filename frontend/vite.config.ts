import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Built to dist/ and served by FastAPI. In dev, `npm run dev` proxies the API
// (and the dashboard WebSocket) to the Python server on :8000.
const API = "http://127.0.0.1:8000";
const routes = ["/health", "/listings", "/skus", "/simulate", "/report", "/negotiate", "/vapi-webhook"];

export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist", emptyOutDir: true },
  server: {
    proxy: {
      ...Object.fromEntries(routes.map((r) => [r, { target: API, changeOrigin: true }])),
      "/dashboard": { target: API.replace("http", "ws"), ws: true },
    },
  },
});
