import {test} from 'node:test';
import assert from 'node:assert/strict';
import {amountClass, estimate} from '../docs/analysis.js';
import {labels, translate} from '../docs/i18n.js';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';

test('cloud amount boundaries', () => {
  assert.deepEqual([0, .1, .1001, .3, .3001, .7, .7001, 1].map(amountClass), [0,0,1,1,2,2,3,3]);
});

test('English defaults and complete static translations', () => {
  const html = readFileSync(new URL('../docs/index.html', import.meta.url), 'utf8');
  assert.match(html, /<html lang="en">/);
  for (const match of html.matchAll(/data-i18n(?:-aria)?="([^"]+)"/g)) {
    for (const locale of ['en', 'zh']) assert.ok(translate(locale, match[1]).length);
  }
  for (const locale of ['en', 'zh']) {
    assert.equal(labels[locale].type.length, 6);
    assert.equal(labels[locale].amount.length, 4);
  }
});

test('localized model progress and scores', () => {
  assert.equal(translate('en', 'downloading', {task: 'cloud type', progress: '50%'}), 'Downloading cloud type model 50% · Your photo stays local');
  assert.equal(translate('zh', 'score', {value: '75.0'}), '得分 75.0%');
});

test('published model checksums, metrics and bilingual README agree', () => {
  const card = JSON.parse(readFileSync(new URL('../docs/models/model-card.json', import.meta.url), 'utf8'));
  const readmes = ['README.md', 'README.zh-CN.md'].map(name => readFileSync(new URL(`../${name}`, import.meta.url), 'utf8'));
  for (const task of ['cloud_type', 'cloud_amount']) {
    const entry = card[task];
    const bytes = readFileSync(new URL(`../docs/models/${task}.onnx`, import.meta.url));
    assert.equal(createHash('sha256').update(bytes).digest('hex'), entry.sha256);
    assert.equal(bytes.length, entry.download_bytes);
    const count = entry.confusion_matrix.flat().reduce((a, b) => a + b, 0);
    const correct = entry.confusion_matrix.reduce((total, row, index) => total + row[index], 0);
    assert.equal(count, entry.split_counts.test);
    assert.equal(correct / count, entry.test_accuracy);
    for (const readme of readmes) {
      assert.ok(readme.includes(`${(entry.test_accuracy * 100).toFixed(1)}%`));
      assert.ok(readme.includes(`${(entry.pytorch_test_accuracy * 100).toFixed(1)}%`));
    }
  }
});
test('blue sky, neutral clouds and empty mask', () => {
  assert.equal(estimate(new Uint8ClampedArray([30,120,220,255]), 1, 1, -.1, false).ratio, 0);
  assert.equal(estimate(new Uint8ClampedArray([230,230,230,255]), 1, 1, -.1, false).ratio, 1);
  assert.equal(estimate(new Uint8ClampedArray([0,0,0,255]), 1, 1, -.1, false).ratio, null);
});
