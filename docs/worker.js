let ort, typeSession, amountSession;

async function loadModel(name) {
  const url = new URL(`./models/${name}.onnx`, import.meta.url).href;
  let cache;
  try { cache = await caches.open('allsky-models-v2'); } catch {}
  let response = cache && await cache.match(url);
  if (!response) {
    self.postMessage({status: `正在下载${name === 'cloud_type' ? '云类型' : '云量'}模型 · 照片仍留在本地`});
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
        self.postMessage({status: `下载${name === 'cloud_type' ? '云类型' : '云量'}模型 ${total ? Math.round(loaded / total * 100) + '%' : (loaded / 1024 / 1024).toFixed(1) + ' MB'} · 照片仍留在本地`});
      }
    }
    const blob = new Blob(chunks);
    response = new Response(blob, {headers: {'Content-Type': 'application/octet-stream'}});
    if (cache) try { await cache.put(url, response.clone()); } catch {}
  }
  return ort.InferenceSession.create(await response.arrayBuffer(), {executionProviders: ['wasm']});
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
    self.postMessage({status: '模型已加载，正在本地分析云量与云类型…'});
    const image = new ort.Tensor('float32', event.data.input, [1,3,224,224]);
    const type = await typeSession.run({image}), amount = await amountSession.run({image});
    self.postMessage({results: {type: softmax(type.logits.data), amount: softmax(amount.logits.data)}});
  } catch (error) { self.postMessage({error: error.message}); }
};
