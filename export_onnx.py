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
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output))
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    actual = session.run(None, {'image': dummy.numpy()})[0]
    with torch.inference_mode():
        expected = net(dummy).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
    print(f'ONNX verified: {output.name}')


if __name__ == '__main__':
    main()
