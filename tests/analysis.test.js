import {test} from 'node:test';
import assert from 'node:assert/strict';
import {amountClass, estimate} from '../docs/analysis.js';

test('cloud amount boundaries', () => {
  assert.deepEqual([0, .1, .1001, .3, .3001, .7, .7001, 1].map(amountClass), [0,0,1,1,2,2,3,3]);
});
test('blue sky, neutral clouds and empty mask', () => {
  assert.equal(estimate(new Uint8ClampedArray([30,120,220,255]), 1, 1, -.1, false).ratio, 0);
  assert.equal(estimate(new Uint8ClampedArray([230,230,230,255]), 1, 1, -.1, false).ratio, 1);
  assert.equal(estimate(new Uint8ClampedArray([0,0,0,255]), 1, 1, -.1, false).ratio, null);
});
