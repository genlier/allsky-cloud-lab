let ort, typeSession, amountSession, modelCard;

async function loadModel(name) {
  if (!modelCard) {
    const manifest = await fetch(new URL('./models/model-card.json', import.meta.url), {cache: 'no-store'});
    if (!manifest.ok) throw new Error('Model manifest download failed');
    modelCard = await manifest.json();
  }
  const digest = modelCard[name].sha256;
  if (!/^[a-f0-9]{64}$/.test(digest)) throw new Error('Invalid model checksum');
  const versioned = new URL(`./models/${name}.onnx`, import.meta.url);
  versioned.searchParams.set('v', digest);
  const url = versioned.href;
  let cache;
  try { cache = await caches.open('allsky-models-v3'); } catch {}
  let response = cache && await cache.match(url);
  if (!response) {
    self.postMessage({statusKey: 'downloading', statusParams: {task: name, progress: ''}});
    response = await fetch(url);
    if (!response.ok) throw new Error('Model download failed');
    const reader = response.body.getReader(), total = Number(response.headers.get('content-length'));
    const chunks = []; let loaded = 0, reported = 0;
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      chunks.push(value); loaded += value.byteLength;
      if (loaded - reported > 1024 * 1024) {
        reported = loaded;
        self.postMessage({statusKey: 'downloading', statusParams: {task: name, progress: total ? Math.min(100, Math.round(loaded / total * 100)) + '%' : (loaded / 1024 / 1024).toFixed(1) + ' MB'}});
      }
    }
    const blob = new Blob(chunks);
    response = new Response(blob, {headers: {'Content-Type': 'application/octet-stream'}});
  }
  const bytes = await response.arrayBuffer();
  const actual = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
    value => value.toString(16).padStart(2, '0')).join('');
  if (actual !== digest) {
    if (cache) await cache.delete(url);
    throw new Error('Model checksum mismatch');
  }
  if (cache) try { await cache.put(url, new Response(bytes)); } catch {}
  return ort.InferenceSession.create(bytes, {executionProviders: ['wasm']});
}

function softmax(values) {
  const maximum = Math.max(...values), exp = Array.from(values, value => Math.exp(value - maximum));
  const sum = exp.reduce((a,b) => a+b, 0);
  return exp.map(value => value / sum);
}

self.onmessage = async event => {
  try {
    if (!ort) {
      ort = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.mjs');
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    }
    if (!typeSession) typeSession = await loadModel('cloud_type');
    if (!amountSession) amountSession = await loadModel('cloud_amount');
    self.postMessage({statusKey: 'running'});
    const image = new ort.Tensor('float32', event.data.input, [1,3,224,224]);
    const type = await typeSession.run({image}), amount = await amountSession.run({image});
    self.postMessage({results: {type: softmax(type.logits.data), amount: softmax(amount.logits.data)}});
  } catch (error) { self.postMessage({error: error.message}); }
};
