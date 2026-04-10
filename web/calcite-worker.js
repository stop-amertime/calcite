/**
 * calc(ite) Web Worker — runs the WASM engine off the main thread.
 *
 * Protocol:
 *   Main → Worker:
 *     { type: 'init', css: string }       — parse and compile CSS
 *     { type: 'tick', count: number }      — run N ticks, return output
 *     { type: 'keyboard', key: number }    — update keyboard state
 *
 *   Worker → Main:
 *     { type: 'ready', video: { text, gfx } }
 *         text/gfx: {addr,size,width,height}|null
 *     { type: 'tick-result', stringProperties, screen, gfxBytes, ticks }
 *         screen  : text-mode rendered string (if text mode detected)
 *         gfxBytes: Uint8ClampedArray.buffer of RGBA pixels (if gfx mode)
 *     { type: 'error', message: string }
 */

let engine = null;
let videoRegions = { text: null, gfx: null };

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
        if (!wasmModule) {
          wasmModule = await loadWasm();
        }
        engine = new wasmModule.CalciteEngine(data.css);

        // Detect video regions. The new JSON shape is {text, gfx}; either
        // can be null. If neither is present, fall back to assuming
        // standard DOS text mode at 0xB8000 so simple programs still work.
        const videoJson = engine.detect_video();
        const parsed = JSON.parse(videoJson) || {};
        videoRegions = {
          text: parsed.text || null,
          gfx: parsed.gfx || null,
        };
        if (!videoRegions.text && !videoRegions.gfx) {
          videoRegions.text = { addr: 0xB8000, size: 4000, width: 80, height: 25 };
        }

        self.postMessage({ type: 'ready', video: videoRegions });
        break;
      }

      case 'tick': {
        if (!engine) {
          throw new Error('Engine not initialised — send "init" first');
        }
        engine.tick_batch(data.count || 1);
        const stringProps = JSON.parse(engine.get_string_properties());

        // Read current video mode from BDA (0x0449). This is the runtime
        // source of truth: what INT 10h AH=00h last wrote. The runner uses
        // it to decide which output to show (text vs. canvas).
        const videoMode = engine.get_video_mode();
        const isGfxMode = videoMode === 0x13; // Mode 13h: 320x200x256

        // Text-mode screen: only rendered when not in a graphics mode.
        let screen = null;
        if (!isGfxMode && videoRegions.text) {
          const t = videoRegions.text;
          screen = engine.render_screen(t.addr, t.width, t.height);
        }

        // Graphics-mode framebuffer: only read when in a graphics mode.
        let gfxBytes = null;
        const transfer = [];
        if (isGfxMode && videoRegions.gfx) {
          const g = videoRegions.gfx;
          // read_framebuffer_rgba returns a Uint8Array backed by wasm
          // memory — copy into a new ArrayBuffer we can transfer.
          const wasmView = engine.read_framebuffer_rgba(g.addr, g.width, g.height);
          const buf = new ArrayBuffer(wasmView.length);
          new Uint8Array(buf).set(wasmView);
          gfxBytes = buf;
          transfer.push(buf);
        }

        self.postMessage(
          {
            type: 'tick-result',
            stringProperties: stringProps,
            screen,
            gfxBytes,
            videoMode,
            ticks: data.count || 1,
          },
          transfer,
        );
        break;
      }

      case 'keyboard': {
        if (engine) {
          engine.set_keyboard(data.key || 0);
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
