// Browser test of the whole flow against a running ReSynth server.
// Usage: node tests/browser/flow.mjs <base_url> <clip.wav> <region_start> <region_end> <out_dir> [screenshot_dir]
// Uploads the clip, drags the region onto [start, end], separates stems, keeps the suggested
// stem, waits for the match, downloads patch.vital into out_dir, and takes screenshots.
import { createRequire } from "module";
import fs from "fs";
import path from "path";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const [base, clip, rs, re, outDir, shotDir] = process.argv.slice(2);
const regionStart = parseFloat(rs), regionEnd = parseFloat(re);
fs.mkdirSync(outDir, { recursive: true });
if (shotDir) fs.mkdirSync(shotDir, { recursive: true });
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

async function shot(page, name, fullPage = true) {
  if (!shotDir) return;
  await page.screenshot({ path: path.join(shotDir, name), fullPage });
  log("screenshot", name);
}

const browser = await chromium.launch();
const errors = [];
try {
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 2, acceptDownloads: true });
  const page = await ctx.newPage();
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });

  await page.goto(base);
  await page.fill("#song", "Test Song");
  await page.fill("#artist", "Test Artist");
  await page.fill("#description", "the pulsing main synth chords with a quick pitch blip on each chord");
  await page.fill("#gear", "Serum, Logic Alchemy");
  await page.setInputFiles("#file", clip);
  await shot(page, "1-upload-desktop.png");

  await page.click("#upload-btn");
  await page.waitForSelector("#step-region:not([hidden])");
  // wait for the waveform and its default region
  const regionSel = "#waveform >> css=[part~='region']";
  await page.waitForFunction(() => {
    const host = document.querySelector("#waveform > div");
    return host && host.shadowRoot && host.shadowRoot.querySelector("[part~='region']");
  }, null, { timeout: 30000 });
  log("default region", await page.textContent("#region-info"));

  // drag the region body so it starts at regionStart, then its right edge to regionEnd
  const wave = await page.locator("#waveform").boundingBox();
  const job = await page.evaluate(() => location.hash.split("=")[1]);
  const state = await (await fetch(`${base}/api/jobs/${job}`)).json();
  const D = state.job.duration_s;
  const xAt = (t) => wave.x + (t / D) * wave.width;
  const reg = await page.locator(regionSel).boundingBox();
  const startNow = ((reg.x - wave.x) / wave.width) * D;
  const midY = reg.y + reg.height / 2;
  await page.mouse.move(reg.x + reg.width / 2, midY);
  await page.mouse.down();
  await page.mouse.move(reg.x + reg.width / 2 + (xAt(regionStart) - xAt(startNow)), midY, { steps: 12 });
  await page.mouse.up();
  const reg2 = await page.locator(regionSel).boundingBox();
  await page.mouse.move(reg2.x + reg2.width - 1, midY);
  await page.mouse.down();
  await page.mouse.move(xAt(regionEnd) - 1, midY, { steps: 12 });
  await page.mouse.up();
  const info = await page.textContent("#region-info");
  log("region now", info);
  await shot(page, "2-region-desktop.png");

  await page.click("#region-btn");
  await page.waitForSelector("#step-stems:not([hidden]) .stem, #error:not([hidden])", { timeout: 300000 });
  if (await page.isVisible("#error")) throw new Error("isolate failed: " + (await page.textContent("#error")));
  const suggested = await page.$eval('input[name="stem"]:checked', (r) => r.value);
  log("stems ready, suggested:", suggested);
  await page.locator("#step-stems").scrollIntoViewIfNeeded();
  await shot(page, "3-stems-desktop.png");
  // try one play button (audio element must accept the stem file)
  await page.click(`#stems .stem:has(input[value="${suggested}"]) .play`);
  await page.waitForTimeout(500);

  await page.click("#stem-btn");
  log("processing...");
  await page.waitForSelector("#step-results:not([hidden]) #downloads a, #error:not([hidden])", { timeout: 900000 });
  if (await page.isVisible("#error")) {
    await shot(page, "error.png");
    throw new Error("processing failed: " + (await page.textContent("#error")));
  }
  const label = await page.textContent("#sim-label");
  log("results:", label);
  await page.waitForTimeout(800);
  await shot(page, "4-results-desktop.png");

  const [download] = await Promise.all([page.waitForEvent("download"), page.click("#downloads a.primary")]);
  const target = path.join(outDir, "downloaded_patch.vital");
  await download.saveAs(target);
  log("downloaded", target, fs.statSync(target).size, "bytes");

  // phone width, same job (reload from the URL), light and dark
  const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  const p2 = await phone.newPage();
  await p2.goto(`${base}/#job=${job}`);
  await p2.waitForSelector("#step-results:not([hidden]) #downloads a", { timeout: 30000 });
  await p2.waitForTimeout(800);
  await shot(p2, "5-results-phone.png");
  const phoneDark = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, colorScheme: "dark" });
  const p3 = await phoneDark.newPage();
  await p3.goto(base);
  await p3.waitForTimeout(500);
  await shot(p3, "6-upload-phone-dark.png");

  fs.writeFileSync(path.join(outDir, "flow.json"), JSON.stringify({ job, label, suggested, region: info, errors }, null, 1));
  if (errors.length) { log("page errors:", errors); process.exitCode = 2; }
  log("ok");
} finally {
  await browser.close();
}
