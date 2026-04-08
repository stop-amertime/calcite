#!/usr/bin/env node
// fulldiff.mjs — Exhaustive tick-by-tick comparison of reference emulator vs calcite.
//
// For EVERY tick, compares ALL registers AND all memory writes. Reports everything:
// - What instruction executed (opcode bytes, address)
// - What BOTH emulators produced for every register
// - What memory was written and what values
// - The FIRST divergence with full context
//
// Usage: node tools/fulldiff.mjs --dos [--ticks=N]
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
const port = parseInt(flags.port || '3333');
const BASE = `http://localhost:${port}`;

// --- Reference emulator setup ---
const cssDir = resolve(__dirname, '..', '..', 'i8086-css');
const js8086Source = readFileSync(resolve(__dirname, 'js8086.js'), 'utf-8');
const evalSource = js8086Source.replace("'use strict';", '').replace('let CPU_186 = 0;', 'var CPU_186 = 0;');
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

// Track ALL memory writes in the reference emulator
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

// --- Main ---
async function main() {
  try { await get('/info'); } catch {
    console.error(`Cannot connect to debugger at ${BASE}. Start it first.`);
    process.exit(1);
  }

  await post('/seek', { tick: 0 });
  console.error(`Full diff: ${maxTicks} ticks, ref vs calcite`);

  let prevRefRegs = getRefRegs();
  let divergences = 0;

  for (let tick = 0; tick < maxTicks; tick++) {
    // Snapshot instruction bytes BEFORE stepping
    const flatIP = prevRefRegs.CS * 16 + prevRefRegs.IP;
    const instBytes = [];
    for (let i = 0; i < 8; i++) instBytes.push(refMem[(flatIP + i) & 0xFFFFF]);

    // Step reference
    refWritesThisTick.length = 0;
    cpu.step();
    const refAfter = getRefRegs();

    // Step calcite
    await post('/tick', { count: 1 });
    const calState = await get('/state');
    const cal = calState.registers;

    // Compare ALL registers
    // FLAGS: mask to only check bits that affect program behavior:
    // CF(0), PF(2), ZF(6), SF(7), DF(10), OF(11)
    // Ignore: reserved(1), AF(4), TF(8), IF(9)
    const FLAGS_MASK = 0x0CC5;  // CF|PF|ZF|SF|DF|OF
    const regDiffs = [];
    for (const r of REG_NAMES) {
      if (r === 'FLAGS') {
        if ((refAfter[r] & FLAGS_MASK) !== (cal[r] & FLAGS_MASK)) regDiffs.push(r);
      } else {
        if (refAfter[r] !== cal[r]) regDiffs.push(r);
      }
    }

    // Compare memory writes: check that calcite's memory matches ref at written addresses
    const memDiffs = [];
    for (const w of refWritesThisTick) {
      const calMem = await post('/memory', { addr: w.addr, len: 1 });
      const calVal = calMem.bytes[0];
      if (calVal !== refMem[w.addr]) {
        memDiffs.push({ addr: w.addr, refVal: refMem[w.addr], calVal, old: w.old });
      }
    }

    if (regDiffs.length > 0 || memDiffs.length > 0) {
      divergences++;
      console.log(`\n${'='.repeat(78)}`);
      console.log(`TICK ${tick}  instruction at ${hexAddr(prevRefRegs.CS, prevRefRegs.IP)}`);
      console.log(`  bytes: ${instBytes.map(b => hex(b, 2)).join(' ')}`);
      console.log(`${'─'.repeat(78)}`);

      // Show ALL register states
      console.log('  Register        Reference    Calcite      Match');
      console.log('  ' + '─'.repeat(55));
      for (const r of REG_NAMES) {
        const rv = refAfter[r], cv = cal[r];
        const match = rv === cv ? '  ✓' : '  ✗ DIFF';
        console.log(`  ${r.padEnd(14)}  ${hex(rv).padEnd(12)} ${hex(cv).padEnd(12)} ${match}`);
      }

      // Show memory writes
      if (refWritesThisTick.length > 0) {
        console.log(`\n  Memory writes (${refWritesThisTick.length} bytes):`);
        console.log('  Address     Old   Ref→    Calcite  Match');
        console.log('  ' + '─'.repeat(50));
        for (const w of refWritesThisTick) {
          const calMem = await post('/memory', { addr: w.addr, len: 1 });
          const calVal = calMem.bytes[0];
          const match = calVal === refMem[w.addr] ? '✓' : '✗ DIFF';
          console.log(`  ${hex(w.addr, 6)}    ${hex(w.old, 2)}    ${hex(refMem[w.addr], 2)}      ${hex(calVal, 2)}       ${match}`);
        }
      }

      // Show what changed in reference (register deltas)
      const changed = [];
      for (const r of REG_NAMES) {
        if (prevRefRegs[r] !== refAfter[r]) changed.push(`${r}: ${hex(prevRefRegs[r])}→${hex(refAfter[r])}`);
      }
      if (changed.length) console.log(`\n  Ref changes: ${changed.join(', ')}`);

      console.log(`${'='.repeat(78)}`);

      // Stop after a few non-FLAGS-only divergences
      const hasNonFlagsDiff = regDiffs.some(r => r !== 'FLAGS') || memDiffs.length > 0;
      if (hasNonFlagsDiff) {
        if (divergences >= 5) {
          console.log(`\nStopping after ${divergences} non-FLAGS divergences (${tick + 1} ticks checked).`);
          break;
        }
      }
    }

    prevRefRegs = refAfter;

    if (tick > 0 && tick % 1000 === 0) {
      console.error(`  ${tick} ticks OK...`);
    }
  }

  if (divergences === 0) {
    console.log(`\n✓ ALL ${maxTicks} ticks match perfectly.`);
  } else {
    console.log(`\n${divergences} divergence(s) found in ${maxTicks} ticks.`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
