import { mkdir, copyFile } from "node:fs/promises";
const files = ["index.html", "app.js", "manifest.webmanifest", "sw.js"];
await mkdir("www", { recursive: true });
for (const file of files) await copyFile(file, `www/${file}`);
console.log("Web assets copied to www/");
