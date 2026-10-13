export const AMOUNTS = ['无云', '少云', '多云', '阴天'];
export function amountClass(ratio) {
  return ratio <= .1 ? 0 : ratio <= .3 ? 1 : ratio <= .7 ? 2 : 3;
}
export function estimate(data, width, height, threshold, circular) {
  let valid = 0, clouds = 0, brightness = 0;
  const mask = new Uint8Array(width * height);
  const radius = Math.min(width, height) * .47;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const pixel = y * width + x, i = pixel * 4;
    if (data[i + 3] < 128) continue;
    if (circular && (x - width / 2) ** 2 + (y - height / 2) ** 2 > radius ** 2) continue;
    const [r, g, b] = [data[i], data[i + 1], data[i + 2]];
    // Exclude nearly black lens boundaries from the candidate area.
    if (Math.max(r, g, b) < 16) continue;
    valid++;
    brightness += .299 * r + .587 * g + .114 * b;
    mask[pixel] = (r - b) / (r + b + 1e-6) > threshold ? 2 : 1;
    if (mask[pixel] === 2) clouds++;
  }
  return {ratio: valid ? clouds / valid : null, valid, mask, brightness: valid ? brightness / valid : 0};
}
