import { amountClass, estimate } from './analysis.js';
import { labels, translate, applyLanguage } from './i18n.js';

const $ = id => document.getElementById(id);
const canvas = $('sky'), ctx = canvas.getContext('2d', {willReadFrequently: true});
const source = document.createElement('canvas');
const sourceCtx = source.getContext('2d', {willReadFrequently: true});
let original, result, cloudResults = [], amountResults = [], worker, imageVersion = 0, masked = false, busy = false;
let locale = 'en', imageTitle = 'sampleTitle', qualityError, modelCard;
let status = {key: 'initial', params: {}};
try { if (localStorage.getItem('allsky-language') === 'zh') locale = 'zh'; } catch {}
const t = (key, params) => translate(locale, key, params);

function renderStatus() {
  const params = {...status.params};
  if (params.task) params.task = t(params.task === 'cloud_type' ? 'typeTask' : 'amountTask');
  $('model-status').textContent = t(status.key, params);
}

function setStatus(key, params = {}) {
  status = {key, params}; renderStatus();
}

function renderResults() {
  $('cloud-type').textContent = cloudResults.length ? labels[locale].type[cloudResults[0].index] : t('waiting');
  $('cnn-amount').textContent = amountResults.length ? labels[locale].amount[amountResults[0].index] : t('waiting');
  $('type-score').textContent = cloudResults.length ? t('candidateScore', {value: (cloudResults[0].score * 100).toFixed(1)}) : '';
  $('cnn-amount-score').textContent = amountResults.length ? t('score', {value: (amountResults[0].score * 100).toFixed(1)}) : '';
  $('rankings').replaceChildren();
  for (const item of cloudResults) {
    const row = document.createElement('div'); row.className = 'rank-row';
    const name = document.createElement('span'); name.textContent = labels[locale].type[item.index];
    const track = document.createElement('div'); track.className = 'rank-track';
    const bar = document.createElement('i'); bar.style.width = `${item.score * 100}%`; track.append(bar);
    const score = document.createElement('span'); score.textContent = `${(item.score * 100).toFixed(0)}%`;
    row.append(name, track, score); $('rankings').append(row);
  }
}

function renderLanguage() {
  applyLanguage(locale);
  $('language').value = locale;
  $('image-title').textContent = t(imageTitle);
  $('validation').textContent = modelCard ? t('metrics', {
    type: (modelCard.cloud_type.test_accuracy * 100).toFixed(1),
    amount: (modelCard.cloud_amount.test_accuracy * 100).toFixed(1),
  }) : t('validation');
  renderStatus(); renderResults(); refresh();
}

function refresh() {
  if (!original) return;
  const threshold = Number($('threshold').value);
  $('threshold-value').textContent = threshold.toFixed(2);
  result = estimate(original.data, source.width, source.height, threshold, $('circle').checked);
  const ratio = result.ratio;
  $('coverage').textContent = ratio === null ? '--' : (ratio * 100).toFixed(1);
  $('amount').textContent = ratio === null ? t('empty') : labels[locale].amount[amountClass(ratio)];
  $('scale-marker').style.left = `${(ratio || 0) * 100}%`;
  $('quality').textContent = t(qualityError || (!result.valid ? 'noSky' : result.brightness < 45 ? 'dark' : 'thresholdHint'));
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
  renderResults();
  if (!busy) setStatus('initial');
}

function setImage(image, title) {
  const factor = Math.min(1, 640 / Math.max(image.width, image.height));
  source.width = canvas.width = Math.max(1, Math.round(image.width * factor));
  source.height = canvas.height = Math.max(1, Math.round(image.height * factor));
  sourceCtx.drawImage(image, 0, 0, source.width, source.height);
  original = sourceCtx.getImageData(0, 0, source.width, source.height);
  imageTitle = title; qualityError = undefined;
  $('image-title').textContent = t(imageTitle);
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
  setImage(c, 'sampleTitle');
}

async function upload(file) {
  if (!file) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || file.size > 20 * 1024 * 1024) {
    qualityError = 'invalidFile'; refresh(); return;
  }
  const url = URL.createObjectURL(file);
  try {
    const image = new Image(); image.src = url;
    await image.decode();
    if (image.width * image.height > 40_000_000) throw new Error('large image');
    setImage(image, 'photoTitle');
  } catch {
    qualityError = 'unreadable'; refresh();
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
  setStatus('loading');
  try {
    if (!worker) worker = new Worker(new URL('./worker.js', import.meta.url), {type: 'module'});
    const predictions = await new Promise((resolve, reject) => {
      worker.onmessage = event => {
        const message = event.data;
        if (message.statusKey) setStatus(message.statusKey, message.statusParams);
        if (message.results) resolve(message.results);
        if (message.error) reject(new Error(message.error));
      };
      worker.onerror = () => reject(new Error('Worker failed'));
      worker.postMessage({input}, [input.buffer]);
    });
    if (version !== imageVersion) {
      setStatus('changed'); return;
    }
    cloudResults = predictions.type.map((score, index) => ({index, score})).sort((a,b) => b.score-a.score);
    amountResults = predictions.amount.map((score, index) => ({index, score})).sort((a,b) => b.score-a.score);
    renderResults(); setStatus('complete');
  } catch (error) {
    worker?.terminate(); worker = undefined;
    setStatus('failed');
    console.error('Model inference failed:', error.message);
  } finally { busy = false; $('analyze').disabled = false; }
}

$('upload').addEventListener('change', event => upload(event.target.files[0]));
$('language').addEventListener('change', event => {
  locale = event.target.value === 'zh' ? 'zh' : 'en';
  try { localStorage.setItem('allsky-language', locale); } catch {}
  renderLanguage();
});
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
  const candidates = (items, task) => items.map(({index, score}) => ({class_id: index + 1, name: labels[locale][task][index], score}));
  const payload = {language: locale, cloud_fraction: result.ratio, threshold_amount: result.ratio === null ? null : labels[locale].amount[amountClass(result.ratio)], threshold: Number($('threshold').value), circular_region: $('circle').checked, cloud_type_candidates: candidates(cloudResults, 'type'), cloud_amount_candidates: candidates(amountResults, 'amount'), methods: {fraction: 'red-blue threshold', classification: 'supervised ResNet18; uncalibrated scores'}};
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'cloud-analysis.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
sample();
renderLanguage();
fetch('./models/model-card.json').then(response => response.json()).then(card => {
  modelCard = card; renderLanguage();
}).catch(() => {});
