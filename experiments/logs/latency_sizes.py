"""CPU latency of the stream tagger per model size, as the app runs it (ONNX Runtime, the graph export_onnx exports,
one stream of MAX_T keys: the longest the app feeds), with untrained weights (speed does not depend on them).
One thread (a slow or busy machine) and all threads. Run with the CPU venv: .venv/Scripts/python
    .venv/Scripts/python -u experiments/logs/latency_sizes.py"""
import os
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, ".")
import onnxruntime as ort  # noqa: E402

import nunchi.export_onnx as ex  # noqa: E402
from nunchi.stream_tagger import MAX_T, StreamTagger  # noqa: E402

SIZES = [  # d, heads, layers, ff
    (256, 4, 4, 1024),  # TR4 now
    (320, 4, 4, 1280),
    (384, 6, 4, 1536),
    (256, 4, 6, 1024),
    (384, 6, 6, 1536),
    (512, 8, 4, 2048),
    (512, 8, 6, 2048),
    (768, 12, 6, 3072),
]
if len(sys.argv) > 1:  # sizes as d,heads,layers,ff ...
    SIZES = [(256, 4, 4, 1024)] + [tuple(map(int, s.split(","))) for s in sys.argv[1:]]
words = ["dkssudgktpdy", "so", "rkwkd", "MBCsms", "good", "wlrma", "qjwjsdms", "tjdrhd", "gkftn", "dlTsmsep", "rkqtlek",
         "goTek", "wkfehoTek"]
enc = ex.encode([[(k, [0] * len(k)) for k in words]])
args = [t[:, -MAX_T:] for t in enc[:4]]
T = args[0].shape[1]
feed = {n: a.numpy() for n, a in zip(("ids", "pos", "wpos", "widx"), args)}
print(f"stream of {T} keys (MAX_T {MAX_T}); CPU threads {os.cpu_count()}", flush=True)
print(f"{'d':>4}{'heads':>6}{'layers':>7}{'ff':>6}{'params':>10}{'x now':>7}{'1 thread ms':>13}{'all threads ms':>16}", flush=True)
base = None
tmp = tempfile.mkdtemp()
for d, h, layers, ff in SIZES:
    m = StreamTagger(left=None, right=None, digits=True, d=d, heads=h, layers=layers, ff=ff).eval()
    n = sum(p.numel() for p in m.parameters())
    path = os.path.join(tmp, f"m_{d}_{layers}.onnx")
    B, Tk = torch.export.Dim("batch"), torch.export.Dim("keys", max=MAX_T)
    torch.onnx.export(ex.Graph(m), tuple(args), path, dynamo=True, external_data=False,
                      input_names=["ids", "pos", "wpos", "widx"], output_names=["emissions"],
                      dynamic_shapes=({0: B, 1: Tk},) * 4)
    row = []
    for threads in (1, 0):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        for _ in range(30):
            sess.run(None, feed)
        ts = []
        for _ in range(300):
            t0 = time.perf_counter()
            sess.run(None, feed)
            ts.append(time.perf_counter() - t0)
        row.append(1000 * float(np.median(ts)))
    base = base or n
    print(f"{d:>4}{h:>6}{layers:>7}{ff:>6}{n:>10,}{n / base:>7.1f}{row[0]:>13.2f}{row[1]:>16.2f}", flush=True)
