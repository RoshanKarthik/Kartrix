import { createServer } from "node:http";
import { loadCatalog } from "./catalog.ts";
import { createApp } from "./router.ts";

const app = createApp(loadCatalog());
const port = Number(process.env.PORT ?? 3000);

createServer((req, res) => {
  let raw = "";
  req.on("data", (chunk) => (raw += chunk));
  req.on("end", () => {
    let body: unknown;
    try {
      body = raw ? JSON.parse(raw) : undefined;
    } catch {
      res.writeHead(400, { "Content-Type": "application/json" }).end(JSON.stringify({ error: "invalid JSON" }));
      return;
    }
    const out = app({ method: req.method ?? "GET", path: req.url ?? "/", body });
    res.writeHead(out.status, { "Content-Type": "application/json" }).end(JSON.stringify(out.body));
  });
}).listen(port, "127.0.0.1", () => console.log(`shop-api on http://127.0.0.1:${port}`));
