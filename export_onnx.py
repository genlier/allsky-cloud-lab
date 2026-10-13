import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from cloud_core import model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--quantize-weights', action='store_true')
    args = parser.parse_args()
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=True)
    net = model(checkpoint['model'], len(checkpoint['labels']))
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    dummy = torch.randn(1, 3, checkpoint['img_size'], checkpoint['img_size'])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, dummy, str(output), input_names=['image'], output_names=['logits'],
                      opset_version=17, dynamo=False)
    graph = onnx.load(str(output))
    graph.doc_string = ''
    graph.graph.doc_string = ''
    del graph.metadata_props[:]
    for node in graph.graph.node:
        node.doc_string = ''
    for tensor in list(graph.graph.initializer) + list(graph.graph.input) + list(graph.graph.output) + list(graph.graph.value_info):
        tensor.doc_string = ''
    if args.quantize_weights:
        original = list(graph.graph.initializer)
        del graph.graph.initializer[:]
        dequantizers = []
        for tensor in original:
            array = onnx.numpy_helper.to_array(tensor)
            if array.dtype != np.float32 or array.ndim < 2:
                graph.graph.initializer.append(tensor)
                continue
            axes = tuple(range(1, array.ndim))
            scale = np.maximum(np.max(np.abs(array), axis=axes) / 127, 1e-8).astype(np.float32)
            quantized = np.clip(np.round(array / scale.reshape((-1,) + (1,) * (array.ndim - 1))), -127, 127).astype(np.int8)
            names = [tensor.name + suffix for suffix in ['__int8', '__scale', '__zero']]
            for value, name in zip([quantized, scale, np.zeros_like(scale, dtype=np.int8)], names):
                graph.graph.initializer.append(onnx.numpy_helper.from_array(value, name))
            dequantizers.append(onnx.helper.make_node('DequantizeLinear', names, [tensor.name], axis=0))
        original_nodes = list(graph.graph.node)
        del graph.graph.node[:]
        graph.graph.node.extend(dequantizers + original_nodes)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output))
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    actual = session.run(None, {'image': dummy.numpy()})[0]
    with torch.inference_mode():
        expected = net(dummy).numpy()
    if args.quantize_weights:
        if not np.isfinite(actual).all():
            raise ValueError('Nonfinite quantized model output')
        print(f'max_logit_difference={np.max(np.abs(actual-expected)):.4f}')
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
    print(f'ONNX verified: {output.name}')


if __name__ == '__main__':
    main()
