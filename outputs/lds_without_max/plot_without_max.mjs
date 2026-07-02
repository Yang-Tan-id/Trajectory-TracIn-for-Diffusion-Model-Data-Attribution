import fs from "node:fs/promises";
import sharp from "sharp";

const csv = await fs.readFile("lds_results_without_max_score.csv", "utf8");
const lines = csv.trim().split(/\r?\n/);
const headers = lines[0].split(",");
const xi = headers.indexOf("pred_sum_tau");
const yi = headers.indexOf("true_f");
const points = lines.slice(1).map(line => {
  const cells = line.split(",");
  return { x: Number(cells[xi]), y: Number(cells[yi]) };
});

function ranks(values) {
  const sorted = values.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const result = Array(values.length);
  for (let i = 0; i < sorted.length;) {
    let j = i + 1;
    while (j < sorted.length && sorted[j].v === sorted[i].v) j++;
    const rank = (i + 1 + j) / 2;
    for (let k = i; k < j; k++) result[sorted[k].i] = rank;
    i = j;
  }
  return result;
}
function correlation(a, b) {
  const ma = a.reduce((s, v) => s + v, 0) / a.length;
  const mb = b.reduce((s, v) => s + v, 0) / b.length;
  let n = 0, da = 0, db = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] - ma, y = b[i] - mb;
    n += x * y; da += x * x; db += y * y;
  }
  return n / Math.sqrt(da * db);
}

const lds = correlation(ranks(points.map(p => p.x)), ranks(points.map(p => p.y)));
const W = 1227, H = 899;
const left = 120, right = 1215, top = 64, bottom = 795;
const xTicks = [-590400000, -590200000, -590000000, -589800000, -589600000, -589400000, -589200000, -589000000, -588800000];
const yTicks = [4.8, 5.0, 5.2, 5.4, 5.6];
const xmin = -590500000, xmax = -588750000, ymin = 4.64, ymax = 5.67;
const sx = x => left + (x - xmin) / (xmax - xmin) * (right - left);
const sy = y => bottom - (y - ymin) / (ymax - ymin) * (bottom - top);

const gridX = xTicks.map(x => `<line x1="${sx(x)}" y1="${top}" x2="${sx(x)}" y2="${bottom}" class="grid"/>
  <line x1="${sx(x)}" y1="${bottom}" x2="${sx(x)}" y2="${bottom + 9}" class="tick"/>
  <text x="${sx(x)}" y="${bottom + 39}" text-anchor="middle" class="tickText">${(x / 1e8).toFixed(3)}</text>`).join("");
const gridY = yTicks.map(y => `<line x1="${left}" y1="${sy(y)}" x2="${right}" y2="${sy(y)}" class="grid"/>
  <line x1="${left - 9}" y1="${sy(y)}" x2="${left}" y2="${sy(y)}" class="tick"/>
  <text x="${left - 18}" y="${sy(y) + 8}" text-anchor="end" class="tickText">${y.toFixed(1)}</text>`).join("");
const dots = points.map(p => `<circle cx="${sx(p.x)}" cy="${sy(p.y)}" r="7.5" fill="#1f77b4" fill-opacity="0.8" stroke="#1f77b4" stroke-width="2"/>`).join("");

const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
<rect width="100%" height="100%" fill="white"/>
<style>
  text { font-family: Arial, Helvetica, sans-serif; fill: #000; }
  .title { font-size: 31px; }
  .axisLabel { font-size: 25px; }
  .tickText { font-size: 24px; }
  .grid { stroke: #b0b0b0; stroke-width: 1.2; opacity: 0.35; }
  .tick { stroke: #000; stroke-width: 1.5; }
</style>
${gridX}${gridY}
<rect x="${left}" y="${top}" width="${right-left}" height="${bottom-top}" fill="none" stroke="#000" stroke-width="2"/>
${dots}
<text x="${(left+right)/2}" y="48" text-anchor="middle" class="title">LDS=${lds.toFixed(4)} (${(lds*100).toFixed(2)}%)</text>
<text x="${(left+right)/2}" y="867" text-anchor="middle" class="axisLabel">Predicted sum of attribution scores</text>
<text x="45" y="${(top+bottom)/2}" text-anchor="middle" class="axisLabel" transform="rotate(-90 45 ${(top+bottom)/2})">True counterfactual f</text>
<text x="${right}" y="${bottom+68}" text-anchor="end" class="tickText">1e8</text>
</svg>`;

await fs.writeFile("lds_scatter_without_max_score.svg", svg);
await sharp(Buffer.from(svg)).png().toFile("lds_scatter_without_max_score.png");
console.log(`LDS=${lds.toFixed(12)}, points=${points.length}`);
