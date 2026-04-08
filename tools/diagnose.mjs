#!/usr/bin/env node
// diagnose.mjs — Property-level conformance diagnosis.
//
// Runs the reference 8086 emulator (js8086) alongside calcite's HTTP debugger,
// compares register state tick-by-tick, and at the first divergence digs into
// CSS property-level diagnostics to show *why* the tick diverged.
//
// Prerequisites: start the debugger first:
//   cargo run --release -p calcite-debugger -- -i <file.css>
//
// Usage:
//   node tools/diagnose.mjs <program.com> <bios.bin> [--ticks=N] [--port=3333]
//
// The tool:
//  1. Steps both emulators tick by tick, comparing registers
//  2. At first divergence, queries calcite's property state via HTTP
//  3. Derives expected CSS property values from reference state + memory
//  4. Diffs properties to pinpoint the root cause

import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// --- Args ---
const args = process.argv.slice(2);
const positional = args.filter(a => !a.startsWith('--'));
const flags = Object.fromEntries(
  args.filter(a => a.startsWith('--')).map(a => {
    const [k, v] = a.split('=');
    return [k.replace(/^--/, ''), v ?? 'true'];
  })
);

const dosMode = 'dos' in flags;

if (!dosMode && positional.length < 2) {
  console.error('Usage: node tools/diagnose.mjs <program.com> <bios.bin> [--ticks=N] [--port=3333]');
  console.error('       node tools/diagnose.mjs --dos [--ticks=N] [--port=3333]');
  process.exit(1);
}

const maxTicks = parseInt(flags.ticks || '500');
const port = parseInt(flags.port || '3333');
const BASE = `http://localhost:${port}`;

// --- Reference emulator setup ---
const js8086Source = readFileSync(resolve(__dirname, 'js8086.js'), 'utf-8');
const evalSource = js8086Source.replace("'use strict';", '').replace('let CPU_186 = 0;', 'var CPU_186 = 0;');
const Intel8086 = new Function(evalSource + '\nreturn Intel8086;')();

const memory = new Uint8Array(1024 * 1024);
let initialRegs;

if (dosMode) {
  // DOS boot mode: kernel at 0x600, disk at 0xD0000, BIOS at 0xF0000
  const cssDir = resolve(__dirname, '..', '..', 'i8086-css');
  const biosBin = readFileSync(resolve(cssDir, 'bios-dos.bin'));
  const kernelBin = readFileSync(resolve(cssDir, 'dos', 'bin', 'kernel.sys'));
  const diskBin = readFileSync(resolve(cssDir, 'dos', 'disk.img'));

  for (let i = 0; i < kernelBin.length; i++) memory[0x600 + i] = kernelBin[i];
  for (let i = 0; i < diskBin.length && 0xD0000 + i < memory.length; i++) memory[0xD0000 + i] = diskBin[i];
  for (let i = 0; i < biosBin.length; i++) memory[0xF0000 + i] = biosBin[i];

  // Get bios_init offset from listing
  let biosInitOffset = 0x37c;
  try {
    const lst = readFileSync(resolve(cssDir, 'bios-dos.lst'), 'utf-8');
    const lines = lst.split('\n');
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].includes('bios_init:')) {
        const m = lines[i + 1]?.match(/([0-9A-Fa-f]{8})/);
        if (m) biosInitOffset = parseInt(m[1], 16);
        break;
      }
    }
  } catch {}

  initialRegs = { cs: 0xF000, ip: biosInitOffset, ss: 0, sp: 0xFFF8, ds: 0, es: 0 };
  console.error(`DOS mode: BIOS init at F000:${biosInitOffset.toString(16)}`);
} else {
  // Simple BIOS mode: .COM at 0x100, BIOS handlers at 0xF0000
  const comBin = readFileSync(resolve(positional[0]));
  const biosBin = readFileSync(resolve(positional[1]));

  for (let i = 0; i < comBin.length; i++) memory[0x100 + i] = comBin[i];
  for (let i = 0; i < biosBin.length; i++) memory[0xF0000 + i] = biosBin[i];

  const BIOS_SEG = 0xF000;
  const handlers = { 0x10: 0x0000, 0x16: 0x0155, 0x1A: 0x0190, 0x20: 0x0232, 0x21: 0x01A9 };
  for (const [intNum, off] of Object.entries(handlers)) {
    const addr = parseInt(intNum) * 4;
    memory[addr] = off & 0xFF;
    memory[addr + 1] = (off >> 8) & 0xFF;
    memory[addr + 2] = BIOS_SEG & 0xFF;
    memory[addr + 3] = (BIOS_SEG >> 8) & 0xFF;
  }

  initialRegs = { cs: 0, ip: 0x0100, ss: 0, sp: 0x05F8, ds: 0, es: 0 };
}

const cpu = Intel8086(
  (addr, val) => { memory[addr & 0xFFFFF] = val & 0xFF; },
  (addr) => memory[addr & 0xFFFFF],
);
cpu.reset();
cpu.setRegs(initialRegs);

function refState() {
  const r = cpu.getRegs();
  return {
    AX: (r.ah << 8) | r.al, CX: (r.ch << 8) | r.cl,
    DX: (r.dh << 8) | r.dl, BX: (r.bh << 8) | r.bl,
    SP: r.sp, BP: r.bp, SI: r.si, DI: r.di,
    IP: r.ip, ES: r.es, CS: r.cs, SS: r.ss, DS: r.ds, FLAGS: r.flags,
  };
}

// Read a 16-bit word from reference memory (little-endian)
function refRead16(addr) {
  addr = addr & 0xFFFFF;
  return memory[addr] | (memory[addr + 1] << 8);
}

// --- HTTP helpers ---
async function post(path, body) {
  const resp = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return resp.json();
}

async function get(path) {
  const resp = await fetch(`${BASE}${path}`);
  return resp.json();
}

// --- Calcite helpers ---
async function calciteTick() {
  return post('/tick', { count: 1 });
}

async function calciteState() {
  return get('/state');
}

async function calciteMemory(addr, len) {
  return post('/memory', { addr, len });
}

async function calciteSeek(tick) {
  return post('/seek', { tick });
}

// --- x86 decode helpers (derived from reference state, not x86 knowledge in calcite!) ---
// These derive what CSS *should* compute, by looking at what the reference emulator sees.

function deriveExpectedProperties(prevRefRegs, refRegs, prevRefMemory) {
  // prevRefRegs = registers BEFORE this tick's instruction executed
  // refRegs = registers AFTER this tick's instruction executed
  // prevRefMemory = memory state before instruction
  const expected = {};
  const ip = prevRefRegs.IP;
  const inst0 = prevRefMemory[ip & 0xFFFFF];
  expected['--inst0_byte'] = inst0;

  // Derive what registers changed
  const REG_NAMES = ['AX', 'CX', 'DX', 'BX', 'SP', 'BP', 'SI', 'DI', 'IP', 'ES', 'CS', 'SS', 'DS', 'FLAGS'];
  const REG_ADDR = { AX: -1, CX: -2, DX: -3, BX: -4, SP: -5, BP: -6, SI: -7, DI: -8, IP: -9, ES: -10, CS: -11, SS: -12, DS: -13, FLAGS: -14 };
  expected['_changed_regs'] = {};
  for (const r of REG_NAMES) {
    if (prevRefRegs[r] !== refRegs[r]) {
      expected['_changed_regs'][r] = { from: prevRefRegs[r], to: refRegs[r], addr: REG_ADDR[r] };
    }
  }

  // Derive SP delta (moveStack)
  expected['_sp_delta'] = refRegs.SP - prevRefRegs.SP;

  // Derive expected IP/CS
  expected['_expected_IP'] = refRegs.IP;
  expected['_expected_CS'] = refRegs.CS;
  expected['_expected_flat_IP'] = refRegs.CS * 16 + refRegs.IP;

  return expected;
}

// --- Main ---
const REG_NAMES = ['AX', 'CX', 'DX', 'BX', 'SP', 'BP', 'SI', 'DI', 'IP', 'ES', 'CS', 'SS', 'DS'];

async function main() {
  // Verify debugger is running
  try {
    await get('/info');
  } catch {
    console.error(`ERROR: Cannot connect to calcite debugger at ${BASE}`);
    console.error('Start it first: cargo run --release -p calcite-debugger -- -i <file.css>');
    process.exit(1);
  }

  // Reset calcite to tick 0
  await calciteSeek(0);

  console.error(`Comparing reference emulator vs calcite debugger for up to ${maxTicks} ticks...`);

  let prevRefRegs = {
    AX: 0, CX: 0, DX: 0, BX: 0,
    SP: initialRegs.sp, BP: 0, SI: 0, DI: 0,
    IP: initialRegs.ip || 0,
    ES: 0, CS: initialRegs.cs || 0, SS: 0, DS: 0, FLAGS: 0,
  };
  // Snapshot memory before first tick
  let prevRefMemory = new Uint8Array(memory);

  for (let tick = 0; tick < maxTicks; tick++) {
    // Step reference
    const memBefore = tick === 0 ? prevRefMemory : new Uint8Array(memory);
    cpu.step();
    const ref = refState();

    // Step calcite
    await calciteTick();
    const calState = await calciteState();
    const cal = calState.registers;

    // Compare registers
    const diffs = [];
    for (const r of REG_NAMES) {
      if (ref[r] !== cal[r]) {
        diffs.push({ reg: r, ref: ref[r], cal: cal[r] });
      }
    }

    if (diffs.length > 0) {
      console.log(`\n${'='.repeat(70)}`);
      console.log(`DIVERGENCE at tick ${tick} (calcite tick ${tick + 1})`);
      console.log(`${'='.repeat(70)}`);

      // Show register diff
      console.log(`\nRegister diffs:`);
      for (const d of diffs) {
        console.log(`  ${d.reg}: ref=${d.ref} (0x${d.ref.toString(16)})  calcite=${d.cal} (0x${d.cal.toString(16)})`);
      }

      // Previous state (input to this tick)
      const flatIP = prevRefRegs.CS * 16 + prevRefRegs.IP;
      console.log(`\nInput state (before this tick):`);
      console.log(`  IP=${prevRefRegs.CS.toString(16)}:${prevRefRegs.IP.toString(16)} (flat 0x${flatIP.toString(16)})  SP=${prevRefRegs.SP}`);
      const ip = flatIP;
      const instBytes = [];
      for (let i = 0; i < 8; i++) instBytes.push(memBefore[(ip + i) & 0xFFFFF]);
      console.log(`  Bytes at IP: ${instBytes.map(b => b.toString(16).padStart(2, '0')).join(' ')}`);

      // Derived expected properties
      const expected = deriveExpectedProperties(prevRefRegs, ref, memBefore);
      console.log(`\nReference instruction analysis:`);
      console.log(`  Opcode byte: 0x${expected['--inst0_byte'].toString(16)} (${expected['--inst0_byte']})`);
      console.log(`  SP delta: ${expected['_sp_delta']}`);
      console.log(`  Changed registers:`);
      for (const [r, info] of Object.entries(expected['_changed_regs'])) {
        console.log(`    ${r}: ${info.from} → ${info.to} (reg addr ${info.addr})`);
      }

      // Now get calcite's full property state — this is the key diagnostic
      console.log(`\nCalcite CSS property state at this tick:`);
      const props = calState.properties || {};

      // Key decode properties
      const DECODE_PROPS = [
        '--instId', '--instLen', '--addrJump',
        '--addrDestA', '--addrDestB', '--addrDestC',
        '--addrValA', '--addrValB', '--addrValC',
        '--moveStack', '--isWordWrite', '--jumpCS',
        '--instArg1', '--instArg1Type', '--instArg2', '--instArg2Type',
        '--instArgDest1', '--instArgDest2',
        '--instSetFlags', '--newFlags',
        '--modRm', '--modRm_mod', '--modRm_reg', '--modRm_rm',
        '--modRmLen', '--isModRm',
        '--intVectorFlat', '--intVectorCS', '--intVectorIP', '--ivtAddr',
        '--moveSI', '--moveDI',
      ];

      console.log(`\n  Instruction decode:`);
      for (const p of DECODE_PROPS) {
        if (props[p] !== undefined) {
          console.log(`    ${p} = ${props[p]}`);
        }
      }

      // Cross-reference: check if addrDest channels match what reference expects
      console.log(`\n  Cross-reference (expected vs actual dest/val):`);
      const destA = props['--addrDestA'];
      const destB = props['--addrDestB'];
      const destC = props['--addrDestC'];
      const valA = props['--addrValA'];
      const valB = props['--addrValB'];
      const valC = props['--addrValC'];

      const changedRegs = expected['_changed_regs'];
      // For each register that changed, check if any dest channel targets it
      for (const [r, info] of Object.entries(changedRegs)) {
        if (r === 'FLAGS' || r === 'SP') continue; // FLAGS/SP handled separately
        const addr = info.addr;
        let found = false;
        if (destA === addr) { found = true; console.log(`    ${r} (addr ${addr}): destA=${destA} valA=${valA} (expected ${info.to})${valA === info.to ? ' ✓' : ' ✗ WRONG'}`); }
        if (destB === addr) { found = true; console.log(`    ${r} (addr ${addr}): destB=${destB} valB=${valB} (expected ${info.to})${valB === info.to ? ' ✓' : ' ✗ WRONG'}`); }
        if (destC === addr) { found = true; console.log(`    ${r} (addr ${addr}): destC=${destC} valC=${valC} (expected ${info.to})${valC === info.to ? ' ✓' : ' ✗ WRONG'}`); }
        if (!found) {
          // Check if it should have been set via addrJump (for IP)
          if (r === 'IP') {
            const addrJump = props['--addrJump'];
            if (addrJump !== undefined && addrJump !== -1) {
              console.log(`    IP: via addrJump=${addrJump} (expected ${info.to})${addrJump === info.to ? ' ✓' : ' ✗ WRONG'}`);
            } else {
              console.log(`    IP: NO channel targets addr ${addr}, addrJump=${addrJump} — IP will advance by instLen=${props['--instLen']}`);
              const advancedIP = prevRefRegs.IP + (props['--instLen'] || 0);
              console.log(`        IP would be ${advancedIP}, expected ${info.to}`);
            }
          } else if (r === 'CS') {
            const jumpCS = props['--jumpCS'];
            console.log(`    CS: NO channel targets addr ${addr}, jumpCS=${jumpCS} (expected ${info.to})`);
          } else {
            console.log(`    ${r} (addr ${addr}): NO dest channel targets this register!`);
          }
        }
      }

      // Check memory around the stack for context
      if (expected['_sp_delta'] !== 0) {
        const sp = prevRefRegs.SP;
        const ss = prevRefRegs.SS;
        const stackBase = ss * 16 + sp;
        console.log(`\n  Stack context (SS:SP = ${ss}:${sp}, flat ${stackBase}):`);
        const stackRange = Math.min(Math.abs(expected['_sp_delta']) + 4, 16);
        const stackStart = expected['_sp_delta'] < 0 ? stackBase + expected['_sp_delta'] : stackBase;

        // Reference stack
        const refStackBytes = [];
        for (let i = 0; i < stackRange; i++) refStackBytes.push(memory[(stackStart + i) & 0xFFFFF]);
        console.log(`    Ref stack at ${stackStart}: ${refStackBytes.map(b => b.toString(16).padStart(2, '0')).join(' ')}`);

        // Calcite stack
        try {
          const calStack = await calciteMemory(stackStart, stackRange);
          console.log(`    Cal stack at ${stackStart}: ${calStack.hex}`);
          if (calStack.hex !== refStackBytes.map(b => b.toString(16).padStart(2, '0').toUpperCase()).join(' ')) {
            console.log(`    *** STACK CONTENTS DIFFER ***`);
          }
        } catch {}
      }

      // Show what the IP formula would evaluate to
      console.log(`\n  IP computation:`);
      console.log(`    addrDestA=${destA} addrDestB=${destB} addrDestC=${destC}`);
      console.log(`    addrValA=${valA} addrValB=${valB} addrValC=${valC}`);
      console.log(`    addrJump=${props['--addrJump']}  instLen=${props['--instLen']}`);
      if (destA === -9) console.log(`    → IP = addrValA = ${valA}`);
      else if (destB === -9) console.log(`    → IP = addrValB = ${valB}`);
      else if (destC === -9) console.log(`    → IP = addrValC = ${valC}`);
      else if (props['--addrJump'] === -1) console.log(`    → IP = __1IP + instLen = ${prevRefRegs.IP} + ${props['--instLen']} = ${prevRefRegs.IP + (props['--instLen'] || 0)}`);
      else console.log(`    → IP = addrJump = ${props['--addrJump']}`);
      console.log(`    Expected IP: ${ref.IP} (0x${ref.IP.toString(16)})`);
      console.log(`    Got IP:      ${cal.IP} (0x${cal.IP.toString(16)})`);

      console.log(`\n${'='.repeat(70)}`);
      break;
    }

    // Progress
    if (tick > 0 && tick % 100 === 0) {
      console.error(`  ${tick} ticks OK...`);
    }

    prevRefRegs = ref;
    prevRefMemory = memBefore;
  }

  if (maxTicks > 0) {
    const finalState = await calciteState();
    console.error(`\nDone. ${maxTicks} ticks checked.`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
