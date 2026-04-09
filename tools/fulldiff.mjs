#!/usr/bin/env node
// fulldiff.mjs — Find the FIRST divergence between JS reference emulator and calcite.
//
// Compares ALL registers including ALL 16 bits of FLAGS (no masking).
// Handles REP sync (JS ref does entire REP in one step, calcite expands per tick).
// On first divergence: prints previous state, instruction, and full comparison, then stops.
//
// Usage: node tools/fulldiff.mjs [--ticks=N] [--skip=N]
//
// Requires: calcite-debugger running on localhost:3333

import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flags = Object.fromEntries(
  args.filter(a => a.startsWith('--')).map(a => {
    const [k, v] = a.split('=');
    return [k.replace(/^--/, ''), v ?? 'true'];
  })
);
const maxTicks = parseInt(flags.ticks || '500');
const skipTicks = parseInt(flags.skip || '0');
const port = parseInt(flags.port || '3333');
const BASE = `http://localhost:${port}`;

// --- Reference emulator setup ---
const cssDir = resolve(__dirname, '..', '..', 'i8086-css');
const js8086Source = readFileSync(resolve(__dirname, 'js8086.js'), 'utf-8');
const evalSource = js8086Source.replace("'use strict';", '').replace('let CPU_186 = 0;', 'var CPU_186 = 1;');
const Intel8086 = new Function(evalSource + '\nreturn Intel8086;')();

const refMem = new Uint8Array(1024 * 1024);

const biosBin = readFileSync(resolve(cssDir, 'bios-dos.bin'));
const kernelBin = readFileSync(resolve(cssDir, 'dos', 'bin', 'kernel.sys'));
const diskBin = readFileSync(resolve(cssDir, 'dos', 'disk.img'));

for (let i = 0; i < kernelBin.length; i++) refMem[0x600 + i] = kernelBin[i];
for (let i = 0; i < diskBin.length && 0xD0000 + i < refMem.length; i++) refMem[0xD0000 + i] = diskBin[i];
for (let i = 0; i < biosBin.length; i++) refMem[0xF0000 + i] = biosBin[i];

let biosInitOffset = 0x385;
try {
  const lst = readFileSync(resolve(cssDir, 'bios-dos.lst'), 'utf-8');
  for (const line of lst.split('\n')) {
    if (line.includes('bios_init:')) {
      const idx = lst.split('\n').indexOf(line);
      const m = lst.split('\n')[idx + 1]?.match(/([0-9A-Fa-f]{8})/);
      if (m) biosInitOffset = parseInt(m[1], 16);
      break;
    }
  }
} catch {}

const refWritesThisTick = [];
const cpu = Intel8086(
  (addr, val) => {
    addr = addr & 0xFFFFF;
    refWritesThisTick.push({ addr, val: val & 0xFF, old: refMem[addr] });
    refMem[addr] = val & 0xFF;
  },
  (addr) => refMem[addr & 0xFFFFF],
);
cpu.reset();
cpu.setRegs({
  cs: 0xF000, ip: biosInitOffset,
  ss: 0, sp: 0xFFF8, ds: 0, es: 0,
  ah: 0, al: 0, bh: 0, bl: 0, ch: 0, cl: 0, dh: 0, dl: 0,
});

const REG_NAMES = ['AX', 'CX', 'DX', 'BX', 'SP', 'BP', 'SI', 'DI', 'IP', 'CS', 'DS', 'ES', 'SS', 'FLAGS'];

function getRefRegs() {
  const r = cpu.getRegs();
  return {
    AX: (r.ah << 8) | r.al, CX: (r.ch << 8) | r.cl,
    DX: (r.dh << 8) | r.dl, BX: (r.bh << 8) | r.bl,
    SP: r.sp, BP: r.bp, SI: r.si, DI: r.di,
    IP: r.ip, CS: r.cs, DS: r.ds, ES: r.es, SS: r.ss, FLAGS: r.flags,
  };
}

// --- REP detection ---
const STRING_OPS = new Set([0xA4, 0xA5, 0xA6, 0xA7, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF]);
const SEG_PREFIXES = new Set([0x26, 0x2E, 0x36, 0x3E]);

function detectREP(cs, ip) {
  const base = (cs * 16 + ip) & 0xFFFFF;
  let off = 0;
  let hasRep = false;
  for (let i = 0; i < 4; i++) {
    const b = refMem[(base + off) & 0xFFFFF];
    if (b === 0xF2 || b === 0xF3) { hasRep = true; off++; }
    else if (SEG_PREFIXES.has(b)) { off++; }
    else break;
  }
  if (!hasRep) return null;
  const opcode = refMem[(base + off) & 0xFFFFF];
  if (!STRING_OPS.has(opcode)) return null;
  return { opcode };
}

// --- HTTP helpers ---
async function post(path, body) {
  const resp = await fetch(`${BASE}${path}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return resp.json();
}
async function get(path) { return (await fetch(`${BASE}${path}`)).json(); }

function hex(v, w = 4) { return v.toString(16).padStart(w, '0'); }
function hexAddr(cs, ip) { return `${hex(cs)}:${hex(ip)} (${hex(cs * 16 + ip, 5)})`; }

function flagBits(f) {
  const names = ['CF','','PF','','AF','','ZF','SF','TF','IF','DF','OF'];
  return names.map((n, i) => n && (f & (1 << i)) ? n : '').filter(Boolean).join('|') || '(none)';
}

// --- Main ---
async function main() {
  try { await get('/info'); } catch {
    console.error(`Cannot connect to debugger at ${BASE}. Start it first.`);
    process.exit(1);
  }

  await post('/seek', { tick: 0 });
  console.error(`Full diff: up to ${maxTicks} ref ticks, all 16 FLAGS bits, REP-aware`);
  if (skipTicks > 0) console.error(`Skipping first ${skipTicks} ticks...`);

  let prevRefRegs = getRefRegs();
  let calciteTick = 0;

  // Skip phase
  if (skipTicks > 0) {
    for (let t = 0; t < skipTicks; t++) {
      const cxBefore = prevRefRegs.CX;
      const rep = detectREP(prevRefRegs.CS, prevRefRegs.IP);
      cpu.step();
      const refAfter = getRefRegs();
      if (rep && cxBefore > 0) {
        const iters = cxBefore - refAfter.CX;
        if (iters > 0) { await post('/tick', { count: iters }); calciteTick += iters; }
        else { await post('/tick', { count: 1 }); calciteTick++; }
      } else {
        await post('/tick', { count: 1 });
        calciteTick++;
      }
      prevRefRegs = refAfter;
      if (t > 0 && t % 5000 === 0) console.error(`  skipped ${t}...`);
    }
    console.error(`  Skip done. Calcite at tick ${calciteTick}.`);
  }

  for (let tick = 0; tick < maxTicks; tick++) {
    const refTick = skipTicks + tick;
    const flatIP = prevRefRegs.CS * 16 + prevRefRegs.IP;
    const instBytes = [];
    for (let i = 0; i < 8; i++) instBytes.push(refMem[(flatIP + i) & 0xFFFFF]);

    const cxBefore = prevRefRegs.CX;
    const rep = detectREP(prevRefRegs.CS, prevRefRegs.IP);

    refWritesThisTick.length = 0;
    cpu.step();
    const refAfter = getRefRegs();

    let calciteSteps = 1;
    if (rep && cxBefore > 0) {
      const iters = cxBefore - refAfter.CX;
      if (iters > 0) calciteSteps = iters;
    }
    await post('/tick', { count: calciteSteps });
    calciteTick += calciteSteps;
    const calState = await get('/state');
    const cal = calState.registers;

    // Compare ALL registers, ALL FLAGS bits
    const regDiffs = [];
    for (const r of REG_NAMES) {
      if (refAfter[r] !== cal[r]) regDiffs.push(r);
    }

    // Check memory writes
    const memDiffs = [];
    const maxMemChecks = 200;
    const writesSample = refWritesThisTick.length > maxMemChecks
      ? refWritesThisTick.filter((_, i) => i < 100 || i >= refWritesThisTick.length - 100)
      : refWritesThisTick;
    for (const w of writesSample) {
      const calMem = await post('/memory', { addr: w.addr, len: 1 });
      const calVal = calMem.bytes[0];
      if (calVal !== refMem[w.addr]) {
        memDiffs.push({ addr: w.addr, refVal: refMem[w.addr], calVal, old: w.old });
      }
    }

    if (regDiffs.length > 0 || memDiffs.length > 0) {
      const repNote = rep ? ` [REP ${hex(rep.opcode, 2)}, CX: ${cxBefore}→${refAfter.CX}, calcite +${calciteSteps}]` : '';
      console.log(`\n${'='.repeat(78)}`);
      console.log(`FIRST DIVERGENCE at ref tick ${refTick} (calcite tick ${calciteTick})`);
      console.log(`${'='.repeat(78)}`);

      // Previous state (before the instruction that caused divergence)
      console.log(`\n  Before: ${hexAddr(prevRefRegs.CS, prevRefRegs.IP)}${repNote}`);
      console.log(`  Instruction bytes: ${instBytes.map(b => hex(b, 2)).join(' ')}`);
      console.log(`  Pre-FLAGS: ref=${hex(prevRefRegs.FLAGS)} [${flagBits(prevRefRegs.FLAGS)}]`);

      // Full register comparison
      console.log(`\n  Register        Reference    Calcite      Match`);
      console.log('  ' + '─'.repeat(60));
      for (const r of REG_NAMES) {
        const rv = refAfter[r], cv = cal[r];
        const match = rv === cv ? '  ✓' : '  ✗ DIFF';
        let extra = '';
        if (r === 'FLAGS' && rv !== cv) {
          const xor = rv ^ cv;
          extra = `  (diff bits: ${flagBits(xor)})`;
        }
        console.log(`  ${r.padEnd(14)}  ${hex(rv).padEnd(12)} ${hex(cv).padEnd(12)} ${match}${extra}`);
      }

      // FLAGS detail
      if (regDiffs.includes('FLAGS')) {
        console.log(`\n  Ref FLAGS:     ${hex(refAfter.FLAGS)} = ${refAfter.FLAGS.toString(2).padStart(16, '0')} [${flagBits(refAfter.FLAGS)}]`);
        console.log(`  Calcite FLAGS: ${hex(cal.FLAGS)} = ${cal.FLAGS.toString(2).padStart(16, '0')} [${flagBits(cal.FLAGS)}]`);
      }

      // Memory diffs
      if (memDiffs.length > 0) {
        console.log(`\n  Memory mismatches (${memDiffs.length} of ${writesSample.length} checked, ${refWritesThisTick.length} total writes):`);
        for (const d of memDiffs.slice(0, 20)) {
          console.log(`    ${hex(d.addr, 6)}: ref=${hex(d.refVal, 2)} cal=${hex(d.calVal, 2)} (was ${hex(d.old, 2)})`);
        }
      } else if (refWritesThisTick.length > 0) {
        console.log(`\n  Memory: ${refWritesThisTick.length} writes, all match ✓`);
      }

      // Ref deltas
      const changed = [];
      for (const r of REG_NAMES) {
        if (prevRefRegs[r] !== refAfter[r]) changed.push(`${r}: ${hex(prevRefRegs[r])}→${hex(refAfter[r])}`);
      }
      if (changed.length) console.log(`\n  Ref deltas: ${changed.join(', ')}`);

      // Calcite properties for debugging
      const props = calState.properties || {};
      const interestingProps = ['opcode', 'hasREP', 'repType', 'prefixLen', 'mod', 'reg', 'rm', 'ea', 'hasSegOverride'];
      const propLines = interestingProps
        .filter(p => `--${p}` in props)
        .map(p => `${p}=${props[`--${p}`]}`);
      if (propLines.length) console.log(`\n  Calcite props: ${propLines.join(', ')}`);

      console.log(`\n${'='.repeat(78)}`);
      console.log(`\nStopped at first divergence. ${refTick} ref ticks matched before this.`);
      break;
    }

    prevRefRegs = refAfter;

    if (tick > 0 && tick % 1000 === 0) {
      console.error(`  ${tick} ref ticks OK (calcite tick ${calciteTick})...`);
    }
  }

}

main().catch(e => { console.error(e); process.exit(1); });
