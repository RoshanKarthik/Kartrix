// Bundles the extension (Node, CommonJS — what VS Code loads) and the chat view's script (browser, IIFE).
// --tests bundles test/*.test.ts for `node --test`; --watch rebuilds on change; --production minifies.
import { build, context } from "esbuild";
import { readdirSync } from "node:fs";

const args = new Set(process.argv.slice(2));
const production = args.has("--production");

const common = { bundle: true, sourcemap: !production, minify: production, logLevel: "info", target: "es2022" };

const targets = args.has("--tests")
  ? [
      {
        ...common,
        entryPoints: readdirSync("test")
          .filter((f) => f.endsWith(".test.ts"))
          .map((f) => `test/${f}`),
        outdir: "out/test",
        platform: "node",
        format: "cjs",
        external: ["vscode"],
        sourcemap: false,
      },
    ]
  : [
      {
        ...common,
        entryPoints: ["src/extension.ts"],
        outfile: "dist/extension.js",
        platform: "node",
        format: "cjs",
        external: ["vscode"],
      },
      {
        ...common,
        entryPoints: ["webview/main.ts"],
        outfile: "dist/webview.js",
        platform: "browser",
        format: "iife",
      },
    ];

if (args.has("--watch")) {
  for (const options of targets) await (await context(options)).watch();
} else {
  await Promise.all(targets.map((options) => build(options)));
}
