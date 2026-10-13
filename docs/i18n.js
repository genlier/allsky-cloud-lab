const messages = {
  title: ['Allsky Cloud Lab', '观云 / Allsky Cloud Lab'],
  description: ['Explore cloud cover and cloud types locally in your browser.', '上传天空照片，在浏览器本地探索云量与云类型。'],
  brand: ['Sky / Lab', '观云'], source: ['Source ↗', '源码 ↗'], language: ['Display language', '显示语言'],
  eyebrow: ['A SMALL OBSERVATORY IN YOUR BROWSER', '浏览器里的小小观测站'],
  headline: ['Read the sky.\nTrace the clouds.', '把天空，\n读成一幅云图。'],
  lede: ['Upload a sky photo to explore cloud cover and cloud types.\nImage processing and inference happen entirely in your browser.', '上传一张天空照片，观察云覆盖的比例，识别云量与云类型。\n图像处理和模型推理都在你的浏览器里完成。'],
  local: ['Local processing', '本地处理'], coverTag: ['Cloud cover', '云量估计'], cnnTag: ['Two CNN models', '双 CNN 识别'],
  workspace: ['Photo analysis', '照片分析'], canvas: ['Sky image and candidate cloud regions', '天空图像与云候选区域'],
  sampleTitle: ['Synthetic sky · Try it now', '合成天空 · 可直接试用'], photoTitle: ['Local photo · Never uploaded', '本地照片 · 未上传服务器'],
  original: ['Original', '原图'], mask: ['Cloud candidate mask', '云候选掩膜'], upload: ['＋ Upload sky photo', '＋ 上传天空照片'], sample: ['Try synthetic sky', '使用合成示例'],
  formats: ['JPG / PNG / WebP, up to 20 MB per photo. Drag and drop works too.', '支持 JPG / PNG / WebP，单张 ≤ 20 MB。也可以拖入照片。'],
  coverHeading: ['01 / CLOUD COVER', '01 / 云覆盖'], thresholdMethod: ['Color threshold', '颜色阈值'], waiting: ['Awaiting analysis', '等待识别'],
  fraction: ['Candidate cloud fraction in the valid region', '有效区域内的云候选比例'],
  clearScale: ['Clear ≤10%', '无云 ≤10%'], fewScale: ['Few ≤30%', '少云 ≤30%'], cloudyScale: ['Cloudy ≤70%', '多云 ≤70%'], overcastScale: ['Overcast', '阴天'],
  threshold: ['Red–blue difference threshold', '红蓝差值阈值'], circle: ['Circular all-sky region', '全天空圆形区域'], circleHint: ['Uncheck for ordinary sky photos', '普通天空照片可取消'],
  empty: ['No valid region', '无有效区域'], noSky: ['Upload a daytime photo with visible sky.', '请上传有可见天空的日间照片。'],
  dark: ['This image is dark; the daytime color threshold is less reliable here.', '图像偏暗，日间颜色阈值在此图像上可靠性较低。'],
  thresholdHint: ['Lower thresholds count more pixels as cloud candidates.', '阈值可调：数值越低，更多像素被计入云候选。'],
  invalidFile: ['Choose a JPG, PNG or WebP image up to 20 MB.', '请选择 20 MB 以内的 JPG、PNG 或 WebP 图片。'], unreadable: ['The image could not be read or is too large. Try another photo.', '图片无法读取或尺寸过大，请更换照片。'],
  neuralHeading: ['02 / NEURAL CLASSIFICATION', '02 / 神经网络分类'], supervised: ['ResNet18 · Supervised', 'ResNet18 · 监督学习'], cnnAmount: ['CNN cloud amount:', 'CNN 云量：'],
  analyze: ['Load models & classify clouds ↗', '加载模型并识别云量、云类型 ↗'],
  initial: ['First use downloads two models, about 23 MB total. Your browser can cache them.', '首次使用需下载两个模型约 23 MB，之后可使用浏览器缓存。'],
  loading: ['Loading public models. Please wait…', '正在加载公开模型，请稍候…'],
  downloading: ['Downloading {task} model {progress} · Your photo stays local', '正在下载{task}模型 {progress} · 照片仍留在本地'],
  running: ['Models loaded. Analyzing cloud amount and type locally…', '模型已加载，正在本地分析云量与云类型…'], changed: ['The photo changed. Click classify again.', '照片已更换，请重新点击识别。'],
  complete: ['Complete · Both CNNs ran locally. Class scores are uncalibrated.', '识别完成 · 两个 CNN 已在浏览器本地运行。类别得分尚未校准。'],
  failed: ['Model loading or inference failed. Check your connection and retry; threshold analysis remains available.', '模型加载或推理失败。请检查网络后重试；云量阈值估计仍可使用。'],
  score: ['Score {value}%', '得分 {value}%'], candidateScore: ['Candidate score {value}%', '候选得分 {value}%'], typeTask: ['cloud type', '云类型'], amountTask: ['cloud amount', '云量'], download: ['↓ Download analysis JSON', '↓ 下载本次分析 JSON'],
  regionTitle: ['Choose the sky region', '选择天空区域'],
  regionNote: ['Use daytime images dominated by sky. Buildings, trees and sun glare affect the color threshold. The circular region applies only to threshold estimates; CNNs use the full image.', '优先上传以天空为主的日间图像。建筑、树木和太阳眩光会影响颜色阈值。圆形区域仅用于云量阈值估计；CNN 使用完整图像。'],
  methodTitle: ['Make the method visible', '让方法可见'],
  methodNote: ['Pixel coverage uses the normalized red–blue difference. Two ResNet18 models predict cloud amount and cloud type separately. Their softmax scores have not been probability-calibrated.', '像素覆盖比例来自红蓝归一化差值；两个 ResNet18 分别预测云量档位和云类别。类别输出的 softmax 得分尚未做概率校准。'],
  evidenceTitle: ['Keep the evidence in view', '保留证据边界'],
  validation: ['Three-fold cross-validation selects settings on deduplicated training data. Final models use the training and validation pool together. The retained test set was inspected in earlier experiments; other devices and new scenes need separate evaluation.', '去重训练数据上的三折交叉验证选择配置，最终合并训练集与验证集重训。保留测试集在此前实验中已被查看；跨设备与新场景的表现需另行评估。'],
  metrics: ['Final test accuracy of the deployed models: cloud type {type}%, cloud amount {amount}%. Three-fold cross-validation selects settings before refitting. The retained test set was inspected in earlier experiments; these results are a retrospective comparison.', '部署模型的最终测试准确率：云类型 {type}%，云量 {amount}%。三折交叉验证选择配置后合并重训。保留测试集在此前实验中已被查看，这些结果属于回顾性比较。'],
  footer: ['Photos stay local · Models download from this site', '照片留在本地 · 模型从本站下载'],
};

export const labels = {
  en: {type: ['Clear sky', 'Cirrus', 'Stratiform', 'Cumuliform', 'Cumulonimbus', 'Mixed clouds'], amount: ['Clear', 'Few clouds', 'Cloudy', 'Overcast']},
  zh: {type: ['晴空', '卷云', '层状云', '积状云', '积雨云', '混合云'], amount: ['无云', '少云', '多云', '阴天']},
};

export function translate(locale, key, params = {}) {
  const entry = messages[key];
  if (!entry) throw new Error(`Unknown translation: ${key}`);
  return entry[locale === 'zh' ? 1 : 0].replace(/\{(\w+)\}/g, (_, name) => params[name] ?? '');
}

export function applyLanguage(locale, root = document) {
  root.documentElement.lang = locale === 'zh' ? 'zh-CN' : 'en';
  root.title = translate(locale, 'title');
  root.querySelector('meta[name="description"]').content = translate(locale, 'description');
  for (const element of root.querySelectorAll('[data-i18n]')) element.textContent = translate(locale, element.dataset.i18n);
  for (const element of root.querySelectorAll('[data-i18n-aria]')) element.setAttribute('aria-label', translate(locale, element.dataset.i18nAria));
}
