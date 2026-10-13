import { AMOUNTS, amountClass, estimate } from './analysis.js';

const $ = id => document.getElementById(id);
const canvas = $('sky'), ctx = canvas.getContext('2d', {willReadFrequently: true});
const source = document.createElement('canvas');
const sourceCtx = source.getContext('2d', {willReadFrequently: true});
let original, result, cloudResults = [], amountResults = [], worker, imageVersion = 0, masked = false, busy = false;
const typeNames = ['晴空', '卷云', '层状云', '积状云', '积雨云', '混合云'];

function refresh() {
  if (!original) return;
  const threshold = Number($('threshold').value);
  $('threshold-value').textContent = threshold.toFixed(2);
  result = estimate(original.data, source.width, source.height, threshold, $('circle').checked);
  const ratio = result.ratio;
  $('coverage').textContent = ratio === null ? '--' : (ratio * 100).toFixed(1);
  $('amount').textContent = ratio === null ? '无有效区域' : AMOUNTS[amountClass(ratio)];
  $('scale-marker').style.left = `${(ratio || 0) * 100}%`;
  $('quality').textContent = !result.valid ? '请上传有可见天空的日间照片。' : result.brightness < 45 ? '图像偏暗，日间颜色阈值在此图像上可靠性较低。' : '阈值可调：数值越低，更多像素被计入云候选。';
  if (!masked) ctx.putImageData(original, 0, 0);
  else {
    const view = new ImageData(new Uint8ClampedArray(original.data), source.width, source.height);
    for (let i = 0; i < result.mask.length; i++) {
      const color = result.mask[i] === 2 ? [184, 212, 105] : result.mask[i] === 1 ? [34, 105, 104] : [226, 232, 224];
      view.data.set([...color, 255], i * 4);
    }
    ctx.putImageData(view, 0, 0);
  }
}

function resetType() {
  imageVersion++;
  cloudResults = [];
  amountResults = [];
  $('cloud-type').textContent = '等待识别';
  $('cnn-amount').textContent = '等待识别';
  $('cnn-amount-score').textContent = '';
  $('type-score').textContent = '';
  $('rankings').replaceChildren();
  if (!busy) $('model-status').textContent = '首次使用需下载两个模型约 90 MB，之后可使用浏览器缓存。';
}

function setImage(image, title) {
  const factor = Math.min(1, 640 / Math.max(image.width, image.height));
  source.width = canvas.width = Math.max(1, Math.round(image.width * factor));
  source.height = canvas.height = Math.max(1, Math.round(image.height * factor));
  sourceCtx.drawImage(image, 0, 0, source.width, source.height);
  original = sourceCtx.getImageData(0, 0, source.width, source.height);
  $('image-title').textContent = title;
  $('dimensions').textContent = `${image.width} × ${image.height}`;
  resetType();
  refresh();
}

function sample() {
  const c = document.createElement('canvas'); c.width = c.height = 512;
  const g = c.getContext('2d');
  const sky = g.createLinearGradient(0, 0, 0, 512);
  sky.addColorStop(0, '#3089b2'); sky.addColorStop(1, '#9bd2df');
  g.fillStyle = sky; g.fillRect(0, 0, 512, 512);
  const clouds = [[160, 155, 90, 30], [215, 139, 65, 43], [270, 161, 92, 29], [380, 328, 70, 21], [335, 316, 44, 32], [86, 355, 86, 26], [40, 342, 50, 33]];
  for (const [x, y, rx, ry] of clouds) {
    const shade = g.createLinearGradient(0, y - ry, 0, y + ry);
    shade.addColorStop(0, '#ffffff'); shade.addColorStop(1, '#dae5e7');
    g.fillStyle = shade; g.beginPath(); g.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2); g.fill();
  }
  setImage(c, '合成天空 · 可直接试用');
}

async function upload(file) {
  if (!file) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || file.size > 20 * 1024 * 1024) {
    $('quality').textContent = '请选择 20 MB 以内的 JPG、PNG 或 WebP 图片。'; return;
  }
  const url = URL.createObjectURL(file);
  try {
    const image = new Image(); image.src = url;
    await image.decode();
    if (image.width * image.height > 40_000_000) throw new Error('large image');
    setImage(image, '本地照片 · 未上传服务器');
  } catch {
    $('quality').textContent = '图片无法读取或尺寸过大，请更换照片。';
  } finally { URL.revokeObjectURL(url); }
}

async function recognize() {
  if (busy) return;
  busy = true; $('analyze').disabled = true;
  const version = imageVersion;
  const resized = document.createElement('canvas'); resized.width = resized.height = 224;
  const context = resized.getContext('2d');
  context.imageSmoothingQuality = 'high';
  context.drawImage(source, 0, 0, 224, 224);
  const pixels = context.getImageData(0, 0, 224, 224).data;
  const input = new Float32Array(3 * 224 * 224);
  const means = [.485, .456, .406], std = [.229, .224, .225];
  for (let i = 0; i < 224 * 224; i++) for (let c = 0; c < 3; c++) input[c * 224 * 224 + i] = (pixels[i * 4 + c] / 255 - means[c]) / std[c];
  $('model-status').textContent = '正在加载公开模型，请稍候…';
  try {
    if (!worker) worker = new Worker(new URL('./worker.js', import.meta.url), {type: 'module'});
    const predictions = await new Promise((resolve, reject) => {
      worker.onmessage = event => {
        const message = event.data;
        if (message.status) $('model-status').textContent = message.status;
        if (message.results) resolve(message.results);
        if (message.error) reject(new Error(message.error));
      };
      worker.onerror = () => reject(new Error('Worker failed'));
      worker.postMessage({input}, [input.buffer]);
    });
    if (version !== imageVersion) {
      $('model-status').textContent = '照片已更换，请重新点击识别。'; return;
    }
    cloudResults = predictions.type.map((score, i) => ({name: typeNames[i], score})).sort((a,b) => b.score-a.score);
    amountResults = predictions.amount.map((score, i) => ({name: AMOUNTS[i], score})).sort((a,b) => b.score-a.score);
    $('cnn-amount').textContent = amountResults[0].name;
    $('cnn-amount-score').textContent = `得分 ${(amountResults[0].score * 100).toFixed(1)}%`;
    $('cloud-type').textContent = cloudResults[0].name;
    $('type-score').textContent = `候选得分 ${(cloudResults[0].score * 100).toFixed(1)}%`;
    $('rankings').replaceChildren();
    for (const item of cloudResults) {
      const row = document.createElement('div'); row.className = 'rank-row';
      const name = document.createElement('span'); name.textContent = item.name;
      const track = document.createElement('div'); track.className = 'rank-track';
      const bar = document.createElement('i'); bar.style.width = `${item.score * 100}%`; track.append(bar);
      const score = document.createElement('span'); score.textContent = `${(item.score * 100).toFixed(0)}%`;
      row.append(name, track, score); $('rankings').append(row);
    }
    $('model-status').textContent = '识别完成 · 两个 CNN 已在浏览器本地运行。类别得分尚未校准。';
  } catch (error) {
    worker?.terminate(); worker = undefined;
    $('model-status').textContent = '模型加载或推理失败。请检查网络后重试；云量阈值估计仍可使用。';
    console.error('Model inference failed:', error.message);
  } finally { busy = false; $('analyze').disabled = false; }
}

$('upload').addEventListener('change', event => upload(event.target.files[0]));
$('sample').addEventListener('click', sample);
$('threshold').addEventListener('input', refresh);
$('circle').addEventListener('change', refresh);
$('original').addEventListener('click', () => {masked = false; $('original').classList.add('selected'); $('mask').classList.remove('selected'); refresh();});
$('mask').addEventListener('click', () => {masked = true; $('mask').classList.add('selected'); $('original').classList.remove('selected'); refresh();});
$('analyze').addEventListener('click', recognize);
const dropzone = $('dropzone');
dropzone.addEventListener('dragover', event => {event.preventDefault(); dropzone.classList.add('dragover');});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', event => {event.preventDefault(); dropzone.classList.remove('dragover'); upload(event.dataTransfer.files[0]);});
$('download').addEventListener('click', () => {
  const payload = {cloud_fraction: result.ratio, threshold_amount: result.ratio === null ? null : AMOUNTS[amountClass(result.ratio)], threshold: Number($('threshold').value), circular_region: $('circle').checked, cloud_type_candidates: cloudResults, cloud_amount_candidates: amountResults, methods: {fraction: 'red-blue threshold', classification: 'supervised ResNet18; uncalibrated scores'}};
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'cloud-analysis.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
sample();
fetch('./models/model-card.json').then(response => response.json()).then(card => {
  $('validation').textContent = `去重后随机划分的最终测试准确率：云类型 ${(card.cloud_type.test_accuracy * 100).toFixed(1)}%，云量 ${(card.cloud_amount.test_accuracy * 100).toFixed(1)}%。独立验证集选择轮次。这些结果仅对应当前划分，跨设备、新场景效果需另行验证。`;
}).catch(() => {});
