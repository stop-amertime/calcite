/**
 * calc(ite) Web Worker — runs the WASM engine off the main thread.
 *
 * Protocol:
 *   Main → Worker:
 *     { type: 'init', css: string }       — parse and compile CSS
 *     { type: 'tick', count: number }      — run N ticks, return output
 *     { type: 'keyboard', key: number }    — JS-bridge keypress (see note)
 *     { type: 'setFramebufferSAB', sab: SharedArrayBuffer, width, height }
 *       — optional. When set, subsequent tick-results will write the RGBA
 *         framebuffer into the shared buffer and omit the per-frame
 *         transfer. Requires a cross-origin-isolated page.
 *
 *   NOTE on keyboard: the CSS-DOS cabinet has pure-CSS
 *   `&:has(#kb-X:active) { --keyboard: V }` rules that drive --keyboard
 *   in raw Chrome. Calcite-wasm doesn't itself evaluate selectors with
 *   :active, so for the web runner we bridge on-screen buttons through
 *   JS: pointerdown/pointerup post a `press` message here with
 *   `{selector, active}` and the worker calls
 *   `engine.set_pseudo_class_active('active', selector, active)`.
 *   Calcite's input-edge recogniser converts the host-side state into
 *   the cabinet's CSS-declared property value at the next tick. The
 *   pure-CSS path still works when the .css is opened in Chrome directly.
 *
 *   Worker → Main:
 *     { type: 'ready', video: { text, gfx } }
 *         text/gfx: {addr,size,width,height}|null
 *     { type: 'tick-result', stringProperties, screen, gfxBytes, ticks, cycles, videoMode, gfxSAB }
 *         screen  : text-mode rendered string (if text mode detected)
 *         gfxBytes: Uint8ClampedArray.buffer of RGBA pixels (legacy path;
 *                   null when gfxSAB is in use)
 *         gfxSAB  : true when the shared framebuffer was updated this tick
 *                   (pixels are already in the SAB the main thread owns)
 *     { type: 'error', message: string }
 *
 * Sibling module: `./video-modes.mjs` contains the video-mode geometry
 * table and the per-mode decoders/rasterisers (text glyphs, Mode 13h,
 * CGA 0x04). That file is the canonical location — CSS-DOS's bridge
 * worker imports the same module via its dev-server alias
 * (`/calcite/video-modes.mjs`), so adding a new mode means editing ONE
 * file that both renderers already consume.
 */

import { pickMode, decodeCga4, decodeCga2, rasteriseText } from './video-modes.mjs';

let engine = null;
let sharedFramebuffer = null;  // Uint8Array view over the main-thread SAB
let sharedFramebufferGeom = null; // {width, height} of the SAB's pixel canvas

// 8×16 VGA ROM font — 4096 bytes, 256 glyphs × 16 rows, bit 7 = leftmost.
// Sent in via the 'setFont' message at startup. While null we fall back to
// engine.render_screen_html() for text modes, which the non-grid player
// still consumes.
let fontAtlas = null;

// Text-mode rasteriser dedup. A DOS prompt sitting idle produces the
// same text VRAM and cursor position for hundreds of ticks at a time.
// Rasterising the font atlas into 256 KB of RGBA on every one of those
// ticks is pure waste. We hash the VRAM bytes + cursor + blink phase
// on each tick and skip the raster call if nothing that would affect
// the output has changed. Hashing 4000 bytes + 4 state bytes is
// ~0.01 ms, cheaper than the raster by a factor of ~2000.
let lastTextHash = -1;

// Self-running tick loop state. The worker paces itself via an
// EMA-driven batch-size adapter so main doesn't have to send a
// 'tick' message per batch. Main only sends control messages
// (init/pause/resume/frame/keyboard).
let running = false;
const TICK_TARGET_MS = 14;
const TICK_EMA_ALPHA = 0.3;
const TICK_MIN_BATCH = 50;
const TICK_MAX_BATCH = 50000;
let tickBatch = 200;
let tickEma = TICK_TARGET_MS;
let tickCount = 0;
let lastStatsPostMs = 0;

// Self-scheduling loop via MessageChannel. setTimeout(0) can get
// clamped; MessageChannel posts are macrotasks with no clamp and
// still let pending messages (frame requests, keyboard) interleave.
const tickChan = new MessageChannel();
tickChan.port2.onmessage = () => tickLoop();

function tickLoop() {
  if (!running || !engine) return;
  const t0 = performance.now();
  engine.tick_batch(tickBatch);
  const dt = performance.now() - t0;
  tickEma = tickEma * (1 - TICK_EMA_ALPHA) + dt * TICK_EMA_ALPHA;
  const ratio = Math.max(0.5, Math.min(2.0, TICK_TARGET_MS / tickEma));
  tickBatch = Math.max(TICK_MIN_BATCH, Math.min(TICK_MAX_BATCH, Math.round(tickBatch * ratio)));
  tickCount += tickBatch;

  // Post a tiny stats update ~every 100 ms so the main thread can
  // update the readouts without round-tripping per tick.
  const now = performance.now();
  if (now - lastStatsPostMs >= 100) {
    lastStatsPostMs = now;
    self.postMessage({
      type: 'tick-stats',
      cycles: engine.get_state_var('cycleCount') >>> 0,
      videoMode: engine.get_video_mode(),
      requestedVideoMode: engine.get_requested_video_mode(),
      haltCode: engine.get_halt_code(),
      tickBatch,
      tickEma,
    });
  }

  // Yield to the event loop so pending messages (frame requests,
  // keyboard, pause) get processed between batches.
  tickChan.port1.postMessage(0);
}

async function loadWasm() {
  const wasm = await import('./pkg/calcite_wasm.js');
  await wasm.default();
  return wasm;
}

let wasmModule = null;

self.onmessage = async function (event) {
  const { type, ...data } = event.data;

  try {
    switch (type) {
      case 'init': {
        const t0 = performance.now();
        if (!wasmModule) {
          wasmModule = await loadWasm();
        }
        const tWasmLoaded = performance.now();

        const cssBytes = data.css.length;
        engine = new wasmModule.CalciteEngine(data.css);
        const tEngineBuilt = performance.now();

        const timing = {
          cssBytes,
          wasmLoadMs: +(tWasmLoaded - t0).toFixed(1),
          parseCompileMs: +(tEngineBuilt - tWasmLoaded).toFixed(1),
          totalMs: +(tEngineBuilt - t0).toFixed(1),
        };
        console.log('[calcite init]', JSON.stringify(timing));

        self.postMessage({ type: 'ready', timing });
        break;
      }

      case 'start': {
        if (!engine) throw new Error('Engine not initialised');
        if (!running) {
          running = true;
          tickChan.port1.postMessage(0);
        }
        break;
      }
      case 'pause': {
        running = false;
        break;
      }

      case 'frame': {
        // Frame request. Looks up the current video mode in the shared
        // mode table (video-modes.mjs), pulls the corresponding VRAM
        // slice via read_memory_range, decodes with the mode-specific
        // decoder, and ships the RGBA either via SAB (cross-origin-
        // isolated pages) or as a transferable ArrayBuffer fallback.
        //
        // Returns { type:'frame-result', gfxSAB, gfxBytes, gfxChecksum,
        // videoMode, stringProperties?, screen? }. stringProperties and
        // screen are only populated if data.includeText is set — they're
        // non-trivial serialisation costs that the grid player skips.
        if (!engine) {
          throw new Error('Engine not initialised — send "init" first');
        }
        const cycles = engine.get_state_var('cycleCount') >>> 0;
        const videoMode = engine.get_video_mode();
        const mode = pickMode(videoMode);

        let screen = null;
        let stringProps = null;
        if (data.includeText) {
          stringProps = JSON.parse(engine.get_string_properties());
        }

        // Pixel dimensions for the current mode.
        const pxW = mode ? mode.width  : 0;
        const pxH = mode ? mode.height : 0;
        const pixelBytes = pxW * pxH * 4;
        const sabMatches = sharedFramebuffer
                           && sharedFramebufferGeom
                           && sharedFramebufferGeom.width  === pxW
                           && sharedFramebufferGeom.height === pxH;

        // Pick the output buffer: the SAB when its geometry matches the
        // current mode, else a freshly allocated transferable ArrayBuffer.
        // Returned via gfxSAB/gfxBytes below.
        let outRGBA = null;
        let outBuffer = null;   // the transferable ArrayBuffer, when not SAB
        function getOutBuffer() {
          if (sabMatches) return sharedFramebuffer;
          outBuffer = new ArrayBuffer(pixelBytes);
          return new Uint8Array(outBuffer);
        }

        // includeText=true keeps the HTML-string renderer path alive for
        // players that want a quick text fallback while the font atlas
        // is still loading. Independent of the pixel render below.
        if (mode && mode.kind === 'text' && data.includeText && !fontAtlas) {
          screen = engine.render_screen_html(mode.vramAddr, mode.textCols, mode.textRows);
        }

        let rendered = false;
        if (mode && mode.kind === 'text' && fontAtlas) {
          const vramBytes = engine.read_memory_range(
            mode.vramAddr, mode.textCols * mode.textRows * 2);
          const bda = engine.read_memory_range(0x0450, 2);
          // VRAM + cursor + blink hash; skip the raster when nothing
          // changed since last frame AND we're writing into a SAB the
          // main thread already sees. (With a fresh transferable we
          // still have to allocate + fill; no point dedup'ing.)
          const CYCLES_PER_FRAME_DEDUP = 68182;
          const frameBucket = Math.floor(cycles / CYCLES_PER_FRAME_DEDUP);
          const blinkPhases = ((frameBucket & 16) ? 2 : 0) | ((frameBucket & 8) ? 1 : 0);
          let h = 0x811C9DC5 | 0;
          for (let i = 0, n = vramBytes.length; i < n; i++) {
            h = Math.imul(h ^ vramBytes[i], 0x01000193);
          }
          h = Math.imul(h ^ bda[0], 0x01000193);
          h = Math.imul(h ^ bda[1], 0x01000193);
          h = Math.imul(h ^ blinkPhases, 0x01000193);
          h = h >>> 0;
          if (h === lastTextHash && sabMatches) {
            // SAB already carries last frame's pixels. Nothing to do.
            rendered = true;
          } else {
            lastTextHash = h;
            outRGBA = getOutBuffer();
            rasteriseText(
              vramBytes, mode.textCols, mode.textRows, outRGBA, fontAtlas,
              { cycleCount: cycles, cursorCol: bda[0], cursorRow: bda[1],
                cursorEnabled: true, blinkMode: true });
            rendered = true;
          }
        } else if (mode && mode.kind === 'mode13') {
          const wasmView = engine.read_framebuffer_rgba(mode.vramAddr, pxW, pxH);
          outRGBA = getOutBuffer();
          outRGBA.set(wasmView);
          rendered = true;
        } else if (mode && mode.kind === 'cga4') {
          const vram = engine.read_memory_range(mode.vramAddr, 0x4000);
          const palReg = engine.read_memory_range(0x04F3, 1)[0] | 0;
          outRGBA = getOutBuffer();
          decodeCga4(vram, palReg, outRGBA, { mono: !!mode.mono });
          rendered = true;
        } else if (mode && mode.kind === 'cga2') {
          const vram = engine.read_memory_range(mode.vramAddr, 0x4000);
          const palReg = engine.read_memory_range(0x04F3, 1)[0] | 0;
          outRGBA = getOutBuffer();
          decodeCga2(vram, palReg, outRGBA);
          rendered = true;
        }

        let gfxBytes = null;
        let gfxSAB = false;
        let gfxChecksum = 0;
        const transfer = [];
        if (rendered) {
          if (sabMatches) {
            gfxSAB = true;
          } else if (outBuffer) {
            gfxBytes = outBuffer;
            transfer.push(outBuffer);
          }
          // FNV-1a over the framebuffer so main can skip repaint when
          // nothing changed pixel-wise.
          const u32View = gfxSAB
            ? new Uint32Array(sharedFramebuffer.buffer, 0, (sharedFramebuffer.byteLength / 4) | 0)
            : new Uint32Array(gfxBytes);
          let cs = 0x811C9DC5 | 0;
          for (let i = 0, n = u32View.length; i < n; i++) {
            cs = (cs ^ u32View[i]);
            cs = Math.imul(cs, 0x01000193);
          }
          gfxChecksum = cs >>> 0;
        }

        self.postMessage(
          {
            type: 'frame-result',
            stringProperties: stringProps,
            screen,
            gfxBytes,
            gfxSAB,
            gfxChecksum,
            videoMode,
            cycles,
          },
          transfer,
        );
        break;
      }

      case 'setFramebufferSAB': {
        // Accept a SharedArrayBuffer from the main thread. Subsequent
        // ticks will write the framebuffer directly into it. The
        // {width, height} tell us whether the SAB's geometry matches
        // whatever mode the guest is currently in — we only write into
        // the SAB when sizes agree (see the `frame` case's sabMatches).
        //
        // `addr` is accepted and ignored for historical reasons: rendering
        // now dispatches off the mode table's `vramAddr`, not a per-run
        // override. Remove once all callers stop sending it.
        if (data.sab instanceof SharedArrayBuffer) {
          sharedFramebuffer = new Uint8Array(data.sab);
          if (data.width && data.height) {
            sharedFramebufferGeom = { width: data.width, height: data.height };
          }
        } else {
          sharedFramebuffer = null;
          sharedFramebufferGeom = null;
        }
        break;
      }

      case 'readMem': {
        // Debug helper — read an arbitrary byte range and post it back.
        // Useful for the page-side harness to confirm guest state like
        // BDA cursor position without a round trip through tick-result.
        if (engine) {
          const out = engine.read_memory_range(data.addr | 0, data.len | 0);
          self.postMessage({ type: 'readMemResult', addr: data.addr, bytes: Array.from(out) });
        }
        break;
      }

      case 'setFont': {
        // 4096-byte VGA 8×16 ROM font. Cached for the life of the worker.
        if (data.font instanceof Uint8Array && data.font.length === 4096) {
          fontAtlas = data.font;
        } else {
          console.warn('[worker] setFont: expected Uint8Array(4096), got', data.font);
        }
        break;
      }

      case 'press': {
        // {selector: 'kb-X', active: true|false}
        if (engine && data.selector) {
          engine.set_pseudo_class_active('active', data.selector, !!data.active);
        }
        break;
      }

      default:
        console.warn(`calc(ite) worker: unknown message type "${type}"`);
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err.message || String(err) });
  }
};
