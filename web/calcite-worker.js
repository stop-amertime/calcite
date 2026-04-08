/**
 * calc(ite) Web Worker — runs the WASM engine off the main thread.
 *
 * Protocol:
 *   Main → Worker:
 *     { type: 'init', css: string }       — parse and compile CSS
 *     { type: 'tick', count: number }      — run N ticks, return changes
 *     { type: 'keyboard', key: number }    — update keyboard state
 *
 *   Worker → Main:
 *     { type: 'ready', video: {addr,size,width,height}|null }
 *     { type: 'tick-result', changes, stringProperties, screen, ticks }
 *     { type: 'error', message: string }
 */

let engine = null;
let videoConfig = null;

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

        // Auto-detect video memory from CSS structure
        const videoJson = engine.detect_video();
        videoConfig = JSON.parse(videoJson);

        self.postMessage({ type: 'ready', video: videoConfig });
        break;
      }

      case 'tick': {
        if (!engine) {
          throw new Error('Engine not initialised — send "init" first');
        }
        const changesJson = engine.tick_batch(data.count || 1);
        const changes = JSON.parse(changesJson);
        const stringProps = JSON.parse(engine.get_string_properties());

        // Render video screen if video memory was detected
        let screen = null;
        if (videoConfig) {
          screen = engine.render_screen(videoConfig.addr, videoConfig.width, videoConfig.height);
        }

        self.postMessage({
          type: 'tick-result',
          changes,
          stringProperties: stringProps,
          screen,
          ticks: data.count || 1,
        });
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
