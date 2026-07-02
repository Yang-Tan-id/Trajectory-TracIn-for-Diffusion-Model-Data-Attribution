import fs from "node:fs/promises";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = process.env.LDS_INPUT_CSV
  || "/Users/rachelsomething/Downloads/m_50_k_5000_seed_42/lds_results.csv";
const outputDir = process.env.LDS_OUTPUT_DIR
  || "/Users/rachelsomething/Desktop/Clear lab/Trajectory-TracIn-for-Diffusion-Model-Data-Attribution/outputs/lds_without_max";

function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n") { row.push(field.replace(/\r$/, "")); rows.push(row); row = []; field = ""; }
    else field += c;
  }
  if (field.length || row.length) { row.push(field.replace(/\r$/, "")); rows.push(row); }
  return rows;
}

function csvEscape(value) {
  const s = String(value);
  return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

function ranks(values) {
  const sorted = values.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const out = Array(values.length);
  for (let i = 0; i < sorted.length;) {
    let j = i + 1;
    while (j < sorted.length && sorted[j].v === sorted[i].v) j++;
    const avg = (i + 1 + j) / 2;
    for (let k = i; k < j; k++) out[sorted[k].i] = avg;
    i = j;
  }
  return out;
}

function pearson(x, y) {
  const mx = x.reduce((a, b) => a + b, 0) / x.length;
  const my = y.reduce((a, b) => a + b, 0) / y.length;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < x.length; i++) {
    const a = x[i] - mx, b = y[i] - my;
    num += a * b; dx += a * a; dy += b * b;
  }
  return num / Math.sqrt(dx * dy);
}

function spearman(rows, predIndex, trueIndex) {
  return pearson(
    ranks(rows.map(r => Number(r[predIndex]))),
    ranks(rows.map(r => Number(r[trueIndex]))),
  );
}

const text = await fs.readFile(inputPath, "utf8");
const parsed = parseCsv(text);
const headers = parsed[0];
const rawRows = parsed.slice(1).filter(r => r.length === headers.length);
const predIndex = headers.indexOf("pred_sum_tau");
const trueIndex = headers.indexOf("true_f");
const subsetIndex = headers.indexOf("subset_id");
if (predIndex < 0 || trueIndex < 0) throw new Error("Required LDS columns are missing");

const maxValue = Math.max(...rawRows.map(r => Number(r[predIndex])));
const removedRows = rawRows.filter(r => Number(r[predIndex]) === maxValue);
const filteredRows = rawRows.filter(r => Number(r[predIndex]) !== maxValue);
const originalLds = spearman(rawRows, predIndex, trueIndex);
const revisedLds = spearman(filteredRows, predIndex, trueIndex);

const typed = rows => rows.map(r => r.map((v, i) =>
  [0, 1, 2, 4, 5, 6].includes(i) ? Number(v) : v
));

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const filtered = workbook.worksheets.add("Filtered Results");
const original = workbook.worksheets.add("Original Results");

summary.getRange("A1:B8").values = [
  ["LDS recalculation", "Value"],
  ["Removal rule", "Remove maximum pred_sum_tau"],
  ["Removed subset_id", removedRows.map(r => r[subsetIndex]).join(", ")],
  ["Removed pred_sum_tau", maxValue],
  ["Original observations", rawRows.length],
  ["Remaining observations", filteredRows.length],
  ["Original Spearman LDS", originalLds],
  ["Recalculated Spearman LDS", revisedLds],
];
summary.getRange("A1:B1").format = {
  fill: "#1F4E78", font: { bold: true, color: "#FFFFFF" },
  borders: { preset: "outside", style: "thin", color: "#163A5C" },
};
summary.getRange("A2:A8").format.font = { bold: true, color: "#1F1F1F" };
summary.getRange("A7:B8").format.fill = "#E2F0D9";
summary.getRange("B7:B8").format.numberFormat = "0.000000";
summary.getRange("A1:B8").format.autofitColumns();
summary.getRange("A1:A8").format.columnWidth = 28;
summary.showGridLines = false;

for (const [sheet, rows] of [[filtered, filteredRows], [original, rawRows]]) {
  sheet.getRangeByIndexes(0, 0, rows.length + 1, headers.length).values = [headers, ...typed(rows)];
  sheet.getRangeByIndexes(0, 0, 1, headers.length).format = {
    fill: "#4472C4", font: { bold: true, color: "#FFFFFF" },
    borders: { preset: "outside", style: "thin", color: "#2F5597" },
  };
  sheet.freezePanes.freezeRows(1);
  sheet.getRange(`F2:G${rows.length + 1}`).format.numberFormat = "0.000000";
  sheet.getRange(`A1:G${rows.length + 1}`).format.autofitColumns();
  sheet.getRange(`H1:I${rows.length + 1}`).format.columnWidth = 58;
  sheet.showGridLines = false;
}

await fs.mkdir(outputDir, { recursive: true });
const out = await SpreadsheetFile.exportXlsx(workbook);
await out.save(`${outputDir}/lds_results_without_max_score.xlsx`);
const filteredCsv = [headers, ...filteredRows]
  .map(r => r.map(csvEscape).join(","))
  .join("\n") + "\n";
await fs.writeFile(`${outputDir}/lds_results_without_max_score.csv`, filteredCsv);

const inspect = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:B8",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 3,
});
console.log(inspect.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);
const preview = await workbook.render({ sheetName: "Summary", range: "A1:B8", scale: 2 });
await fs.writeFile(`${outputDir}/summary_preview.png`, new Uint8Array(await preview.arrayBuffer()));
console.log(JSON.stringify({ maxValue, removedSubsetIds: removedRows.map(r => r[subsetIndex]), originalLds, revisedLds }));
