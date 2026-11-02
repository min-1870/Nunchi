"""Models by name (TX4, RV0, ...) from models/registry.json: the file, the context it was built for, and the
thresholds tuned for it on the dev sets.

    from nunchi.models import load
    tx4 = load("TX4")            # a StreamTagger on the GPU if there is one, in eval mode
    tx4.split_threshold, tx4.act_threshold
"""
import json

import torch

from nunchi.train_fasttext import MODELS

REGISTRY = MODELS / "registry.json"


def spec(name):
    """The registry entry for a model name."""
    for m in json.loads(REGISTRY.read_text(encoding="utf-8"))["models"]:
        if m["name"] == name:
            return m
    raise KeyError(f"{name} is not in {REGISTRY}")


def load(name, device=None):
    """A stream model by name, with .name, .split_threshold and .act_threshold set (None where untuned:
    the evaluation code then falls back to its defaults)."""
    from nunchi.stream_tagger import StreamTagger
    from nunchi.tagger import device as default_device

    s = spec(name)
    if s.get("kind") != "stream":
        raise ValueError(f"{name} is not a stream model")
    size = dict(zip(("d", "heads", "layers", "ff"), s["size"])) if s.get("size") else {}
    m = StreamTagger(left=s["left"], right=s["right"], digits=s.get("digits", False), **size)
    m.load_state_dict(torch.load(MODELS / s["file"], map_location="cpu"))
    m.name = name
    m.split_threshold = s.get("split_threshold")
    m.act_threshold = s.get("act_threshold")
    return m.to(device or default_device()).eval()
