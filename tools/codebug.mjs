#!/usr/bin/env node
// codebug.mjs — co-execution debugger for calcite vs js8086 reference.
//
// Runs both the JS reference emulator and calcite-debugger in lockstep against
// the same CSS program, exposing a unified HTTP API so you can step, inspect,
// send keyboard input, and diff both sides to find the exact tick at which
// calcite (or more likely, the CSS) diverges from ground truth.
//
// Usage:
//   node tools/codebug.mjs <program.css> [--port=3334] [--calcite-port=3333]
//
// Phase 1 endpoints (this file):
//   GET  /info                  Both sides' metadata, current ticks, agreement flag.
//   POST /step  {count}         Advance both sides by N ticks. Returns divergence info.
//   POST /key   {ascii,scancode|value}
//                               Queue a key; next /step flushes it to BOTH sides.
//   GET  /regs                  Both sides' registers plus a diffs array.
//   GET  /screen                Both sides' VGA text buffers + a side-by-side diff.
//   POST /compare  {memory?}    Diff registers (always) and memory ranges (optional).
//                               memory = [{addr, len}, ...]. Uses each side's memory.
//   POST /seek  {tick}          Reset both sides and advance to `tick`. Expensive!
//                               (JS side has no checkpoints in phase 1 — full replay.)
//   POST /shutdown              Stop the calcite-debugger child and exit.
//
// Design notes:
// - The JS emulator is in-process. calcite-debugger runs as a child process
//   and is driven over its HTTP API on --calcite-port (default 3333).
// - Keyboard routing: both sides' BIOS (gossamer-dos) polls linear 0x500 for
//   keys (word: ASCII in low byte, scancode in high byte). So on each /step,
//   if a key is queued we write it to both sides before executing the batch.
//   The BIOS's INT 16h handler clears 0x500 when the key is consumed.
// - Divergence checking in phase 1 is explicit (/compare or /regs). Phase 2
//   adds /run-until-diverge with auto-bisection.

import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import http from 'http';
const { createServer } = http;

const __dirname = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const positional = args.filter(a => !a.startsWith('--'));
const flags = Object.fromEntries(
  args.filter(a => a.startsWith('--')).map(a => {
    const [k, v] = a.split('=');
    return [k.replace(/^--/, ''), v ?? 'true'];
  })
);

if (positional.length < 1) {
  console.error('Usage: node tools/codebug.mjs <program.css> [--port=3334] [--calcite-port=3333]');
  process.exit(1);
}

const cssPath = resolve(positional[0]);
const port = parseInt(flags.port || '3334');
const calcitePort = parseInt(flags['calcite-port'] || '3333');
const calciteBin = flags['calcite-bin'] || resolve(__dirname, '..', 'target', 'release', 'calcite-debugger.exe');

// ---------------------------------------------------------------------------
// JS reference boot (mirrors ref-dos.mjs)
// ---------------------------------------------------------------------------

const js8086Source = readFileSync(resolve(__dirname, '..', '..', 'CSS-DOS', 'tools', 'js8086.js'), 'utf-8');
const evalSource = js8086Source.replace("'use strict';", '').replace('let CPU_186 = 0;', 'var CPU_186 = 1;');
const Intel8086 = new Function(evalSource + '\nreturn Intel8086;')();

const cssDir = resolve(__dirname, '..', '..', 'CSS-DOS');
const biosBin = readFileSync(resolve(cssDir, 'build', 'gossamer-dos.bin'));
const kernelBin = readFileSync(resolve(cssDir, 'dos', 'bin', 'kernel.sys'));
const diskBin = readFileSync(resolve(cssDir, 'dos', 'disk.img'));

const jsMemory = new Uint8Array(1024 * 1024);
for (let i = 0; i < kernelBin.length; i++) jsMemory[0x600 + i] = kernelBin[i];
for (let i = 0; i < diskBin.length && 0xD0000 + i < jsMemory.length; i++) jsMemory[0xD0000 + i] = diskBin[i];
for (let i = 0; i < biosBin.length; i++) jsMemory[0xF0000 + i] = biosBin[i];

const jsCpu = Intel8086(
  (addr, val) => { jsMemory[addr & 0xFFFFF] = val & 0xFF; },
  (addr) => jsMemory[addr & 0xFFFFF],
);
jsCpu.reset();

// Initial registers are set after calcite-debugger is ready (see initJsCpu below),
// so they match the CSS @property initial-values exactly.
let jsTick = 0;

function jsGetRegs() {
  const r = jsCpu.getRegs();
  return {
    AX: (r.ah << 8) | r.al,
    CX: (r.ch << 8) | r.cl,
    DX: (r.dh << 8) | r.dl,
    BX: (r.bh << 8) | r.bl,
    SP: r.sp, BP: r.bp, SI: r.si, DI: r.di,
    IP: r.ip, CS: r.cs, DS: r.ds, ES: r.es, SS: r.ss, FLAGS: r.flags,
  };
}

function jsStep() {
  jsCpu.step();
  jsTick++;
}

function jsReset() {
  // Re-zero memory and reload BIOS/kernel/disk.
  jsMemory.fill(0);
  for (let i = 0; i < kernelBin.length; i++) jsMemory[0x600 + i] = kernelBin[i];
  for (let i = 0; i < diskBin.length && 0xD0000 + i < jsMemory.length; i++) jsMemory[0xD0000 + i] = diskBin[i];
  for (let i = 0; i < biosBin.length; i++) jsMemory[0xF0000 + i] = biosBin[i];
  jsCpu.reset();
  jsCpu.setRegs({
    cs: 0xF000, ip: biosInitOffset,
    ss: 0, sp: 0xFFF8, ds: 0, es: 0,
    ah: 0, al: 0, bh: 0, bl: 0, ch: 0, cl: 0, dh: 0, dl: 0,
  });
  jsTick = 0;
}

function jsRenderScreen(base = 0xB8000, width = 80, height = 25) {
  // Match calcite's render_screen: printable chars kept, non-printables become spaces.
  // Trim trailing spaces per row, then drop trailing blank lines.
  const rows = [];
  for (let y = 0; y < height; y++) {
    let line = '';
    for (let x = 0; x < width; x++) {
      const a = base + (y * width + x) * 2;
      const ch = jsMemory[a & 0xFFFFF];
      line += ch >= 0x20 && ch < 0x7F ? String.fromCharCode(ch) : ' ';
    }
    rows.push(line.trimEnd());
  }
  while (rows.length && rows[rows.length - 1] === '') rows.pop();
  return rows.join('\n');
}

function jsWriteKey(value) {
  // BIOS polls linear 0x500 (word: ASCII in low byte, scancode in high byte).
  jsMemory[0x500] = value & 0xFF;
  jsMemory[0x501] = (value >> 8) & 0xFF;
}

// ---------------------------------------------------------------------------
// Calcite side — spawn + HTTP client
// ---------------------------------------------------------------------------

let calciteProc = null;

function httpRequest(method, path, body) {
  return new Promise((res, rej) => {
    const data = body == null ? '' : JSON.stringify(body);
    const req = http.request({
      host: '127.0.0.1', port: calcitePort, path, method,
      headers: body == null ? {} : { 'content-type': 'application/json', 'content-length': Buffer.byteLength(data) },
    }, (r) => {
      const chunks = [];
      r.on('data', c => chunks.push(c));
      r.on('end', () => {
        const txt = Buffer.concat(chunks).toString('utf-8');
        if (r.statusCode >= 400) return rej(new Error(`calcite ${path} ${r.statusCode}: ${txt}`));
        try { res(txt ? JSON.parse(txt) : {}); }
        catch { res(txt); }
      });
    });
    req.on('error', rej);
    if (data) req.write(data);
    req.end();
  });
}

// Seed the JS CPU registers from calcite's tick-0 state so they exactly
// match the @property initial-values in the CSS, regardless of which BIOS
// version is currently built.
async function initJsCpu() {
  const state = await httpRequest('GET', '/state');
  const r = state.registers;
  jsCpu.setRegs({
    cs: r.CS, ip: r.IP & 0xFFFF,
    ss: r.SS, sp: r.SP,
    ds: r.DS, es: r.ES,
    ah: (r.AX >> 8) & 0xFF, al: r.AX & 0xFF,
    bh: (r.BX >> 8) & 0xFF, bl: r.BX & 0xFF,
    ch: (r.CX >> 8) & 0xFF, cl: r.CX & 0xFF,
    dh: (r.DX >> 8) & 0xFF, dl: r.DX & 0xFF,
    bp: r.BP, si: r.SI, di: r.DI,
  });
  // FLAGS can't be set via setRegs in js8086 — it starts from cpu.reset() defaults,
  // which is close enough (usually 0x0002, matching the CSS initial value).
  console.error(`[codebug] JS CPU seeded from CSS: CS=${r.CS.toString(16)} IP=${(r.IP&0xFFFF).toString(16)} SP=${r.SP.toString(16)}`);
}

async function waitForCalcite(maxMs = 60000) {
  const deadline = Date.now() + maxMs;
  while (Date.now() < deadline) {
    try {
      await httpRequest('GET', '/info');
      return true;
    } catch {
      await new Promise(r => setTimeout(r, 200));
    }
  }
  throw new Error('calcite-debugger did not come up in time');
}

function startCalcite() {
  console.error(`[codebug] starting calcite-debugger: ${calciteBin} -i ${cssPath} -p ${calcitePort}`);
  calciteProc = spawn(calciteBin, ['-i', cssPath, '-p', String(calcitePort)], {
    stdio: ['ignore', 'inherit', 'inherit'],
  });
  calciteProc.on('exit', (code, signal) => {
    console.error(`[codebug] calcite-debugger exited code=${code} signal=${signal}`);
    calciteProc = null;
  });
}

// ---------------------------------------------------------------------------
// Shared key queue
// ---------------------------------------------------------------------------

const keyQueue = [];

function pushKey(value) {
  keyQueue.push(value & 0xFFFF);
}

// ---------------------------------------------------------------------------
// Register diff helper
// ---------------------------------------------------------------------------

const REG_NAMES = ['AX','CX','DX','BX','SP','BP','SI','DI','IP','ES','CS','SS','DS','FLAGS'];

function diffRegs(jsRegs, calciteRegs) {
  const diffs = [];
  for (const name of REG_NAMES) {
    const j = jsRegs[name] | 0;
    const c = calciteRegs[name] | 0;
    // Normalize 16-bit view for comparison (calcite's IP holds a flat address sometimes).
    if (name === 'IP') {
      // Compare IP as (flat - CS*16) on the calcite side if needed.
      const jIp = j & 0xFFFF;
      const cIp = c & 0xFFFF;
      if (jIp !== cIp) diffs.push({ name, js: jIp, calcite: cIp });
      continue;
    }
    if ((j & 0xFFFF) !== (c & 0xFFFF)) {
      diffs.push({ name, js: j & 0xFFFF, calcite: c & 0xFFFF });
    }
  }
  return diffs;
}

// ---------------------------------------------------------------------------
// Core operations
// ---------------------------------------------------------------------------

async function flushKey() {
  if (keyQueue.length === 0) return null;
  const value = keyQueue.shift();
  jsWriteKey(value);
  await httpRequest('POST', '/key', { value });
  return value;
}

async function stepBoth(count) {
  // Flush one queued key per /step batch — matches how real input behaves
  // (one keypress per poll window).
  const flushed = await flushKey();

  // Calcite side: single HTTP call for the whole batch (cheap).
  const calciteResp = await httpRequest('POST', '/tick', { count });

  // JS side: loop in-process.
  let jsError = null;
  for (let i = 0; i < count; i++) {
    try { jsStep(); }
    catch (e) { jsError = e.message || String(e); break; }
  }

  return {
    ticks_requested: count,
    ticks_js: jsTick,
    ticks_calcite: calciteResp.tick,
    key_flushed: flushed,
    js_error: jsError,
  };
}

async function getBothRegs() {
  const calcite = await httpRequest('GET', '/state');
  const js = jsGetRegs();
  return {
    tick_js: jsTick,
    tick_calcite: calcite.tick,
    js,
    calcite: calcite.registers,
    diffs: diffRegs(js, calcite.registers),
  };
}

// Normalize a screen text for comparison: non-printable → space, rstrip each
// row, drop trailing blank rows. Also maps common CP437 glyphs back to spaces
// since calcite renders CP437 and JS renders ASCII.
function normalizeScreen(text) {
  const rows = text.split('\n').map(row => {
    let out = '';
    for (const ch of row) {
      const cp = ch.codePointAt(0);
      if (cp >= 0x20 && cp < 0x7F) out += ch;
      else out += ' ';
    }
    return out.replace(/\s+$/, '');
  });
  while (rows.length && rows[rows.length - 1] === '') rows.pop();
  return rows.join('\n');
}

async function getBothScreens() {
  const [calciteResp, calciteState] = await Promise.all([
    httpRequest('POST', '/screen', { addr: 0xB8000, width: 80, height: 25 }),
    httpRequest('GET', '/state'),
  ]);
  const jsText = normalizeScreen(jsRenderScreen());
  const calciteText = normalizeScreen(calciteResp.text || '');
  // Side-by-side: zip lines, mark diffs with '|' else ' '.
  const jl = jsText.split('\n');
  const cl = calciteText.split('\n');
  const n = Math.max(jl.length, cl.length);
  const lines = [];
  for (let i = 0; i < n; i++) {
    const a = (jl[i] || '').padEnd(80);
    const b = (cl[i] || '').padEnd(80);
    const mark = a === b ? ' ' : '|';
    lines.push(`${String(i).padStart(2)} ${a}  ${mark}  ${b}`);
  }
  return {
    tick_js: jsTick,
    tick_calcite: calciteState.tick,
    js: jsText,
    calcite: calciteText,
    side_by_side: lines.join('\n'),
    agrees: jsText === calciteText,
  };
}

async function compareMemoryRanges(ranges) {
  // Query calcite for each range, then diff against jsMemory.
  const results = [];
  for (const r of ranges) {
    const calciteResp = await httpRequest('POST', '/memory', { addr: r.addr, len: r.len });
    const cbytes = calciteResp.bytes; // array of u8
    const diffs = [];
    for (let i = 0; i < r.len; i++) {
      const a = r.addr + i;
      const js = jsMemory[a & 0xFFFFF];
      const c = cbytes[i];
      if (js !== c) diffs.push({ addr: a, js, calcite: c });
    }
    results.push({ addr: r.addr, len: r.len, diff_count: diffs.length, diffs });
  }
  return results;
}

async function seekBoth(targetTick) {
  if (targetTick < jsTick) {
    // JS side has no checkpoints in phase 1 — reset and replay.
    jsReset();
  }
  await httpRequest('POST', '/seek', { tick: targetTick });
  while (jsTick < targetTick) {
    try { jsStep(); }
    catch (e) { return { ok: false, error: e.message, tick_js: jsTick }; }
  }
  return { ok: true, tick_js: jsTick };
}

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------

function sendJson(res, code, obj) {
  const body = JSON.stringify(obj, null, 2);
  res.writeHead(code, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

async function readBody(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  const s = Buffer.concat(chunks).toString('utf-8');
  return s ? JSON.parse(s) : {};
}

const server = createServer(async (req, res) => {
  const path = req.url.split('?')[0];
  try {
    if (req.method === 'GET' && path === '/info') {
      const calciteInfo = await httpRequest('GET', '/info').catch(e => ({ error: e.message }));
      return sendJson(res, 200, {
        css: cssPath,
        calcite_port: calcitePort,
        tick_js: jsTick,
        tick_calcite: calciteInfo.current_tick ?? null,
        agreed: (calciteInfo.current_tick ?? -1) === jsTick,
        calcite: calciteInfo,
        key_queue_depth: keyQueue.length,
        endpoints: [
          'GET /info', 'POST /step', 'POST /key', 'GET /regs',
          'GET /screen', 'POST /compare', 'POST /seek', 'POST /shutdown',
        ],
      });
    }

    if (req.method === 'POST' && path === '/step') {
      const body = await readBody(req);
      const count = body.count ?? 1;
      const r = await stepBoth(count);
      return sendJson(res, 200, r);
    }

    if (req.method === 'POST' && path === '/key') {
      const body = await readBody(req);
      let value;
      if (typeof body.value === 'number') {
        value = body.value;
      } else {
        const scan = body.scancode | 0;
        const ascii = body.ascii | 0;
        value = (scan << 8) | (ascii & 0xFF);
      }
      pushKey(value);
      return sendJson(res, 200, { queued: value, depth: keyQueue.length });
    }

    if (req.method === 'GET' && path === '/regs') {
      return sendJson(res, 200, await getBothRegs());
    }

    if (req.method === 'GET' && path === '/screen') {
      return sendJson(res, 200, await getBothScreens());
    }

    if (req.method === 'POST' && path === '/compare') {
      const body = await readBody(req);
      const regs = await getBothRegs();
      const memory = body.memory ? await compareMemoryRanges(body.memory) : [];
      const memDiffCount = memory.reduce((a, m) => a + m.diff_count, 0);
      return sendJson(res, 200, {
        tick_js: jsTick,
        tick_calcite: regs.tick_calcite,
        register_diffs: regs.diffs,
        memory_diffs: memory,
        total_diffs: regs.diffs.length + memDiffCount,
        agrees: regs.diffs.length + memDiffCount === 0,
      });
    }

    if (req.method === 'POST' && path === '/seek') {
      const body = await readBody(req);
      const r = await seekBoth(body.tick | 0);
      return sendJson(res, 200, r);
    }

    if (req.method === 'POST' && path === '/shutdown') {
      sendJson(res, 200, { ok: true });
      await httpRequest('POST', '/shutdown').catch(() => {});
      if (calciteProc) calciteProc.kill();
      setTimeout(() => process.exit(0), 100);
      return;
    }

    sendJson(res, 404, { error: `unknown endpoint ${req.method} ${path}` });
  } catch (e) {
    sendJson(res, 500, { error: e.message || String(e), stack: e.stack });
  }
});

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

startCalcite();
await waitForCalcite();
await initJsCpu();
console.error(`[codebug] calcite-debugger ready on :${calcitePort}`);

server.listen(port, '127.0.0.1', () => {
  console.error(`[codebug] listening on http://localhost:${port}`);
  console.error(`[codebug] try: curl localhost:${port}/info`);
});

process.on('SIGINT', () => {
  console.error('\n[codebug] SIGINT, shutting down');
  if (calciteProc) calciteProc.kill();
  process.exit(0);
});
