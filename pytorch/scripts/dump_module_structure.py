#!/usr/bin/env python3
"""
Dump a precise nn.Module structure tree (schema v2) for JavaCPP StructureModuleBuilder.

Pure state_dict cannot recover param-free layers (ReLU/Dropout p/Softmax/Sigmoid/
Identity/Dice/…). This exporter walks named_children + hypers so Java can rebuild
an exact typed Module tree, then bind weights from a sibling .pth.

Usage:
  # from a live training script:
  from dump_module_structure import dump_structure
  dump_structure(model, "model.structure.json")

  # CLI (needs a pickled full Module — rare; prefer train-hook):
  python3 scripts/dump_module_structure.py --help

Schema v2:
  {
    "version": 2,
    "root": "DSSM",
    "nodes": {
      "": {"kind":"CONTAINER","class_name":"DSSM","children":[...]},
      "user_tower.mlp.2": {"kind":"DROPOUT","class_name":"Dropout","hyper":{"p":0.1}},
      ...
    },
    "parameters": ["...state_dict keys..."],
    "buffers": ["...buffer keys..."]
  }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from torch import nn
except ImportError as e:
    raise SystemExit("torch required: " + str(e))


# ---------------------------------------------------------------------------
# Kind classification
# ---------------------------------------------------------------------------

def _cls_name(m: nn.Module) -> str:
    return type(m).__name__


def _classify(m: nn.Module) -> Tuple[str, Dict[str, Any]]:
    """Return (kind, hyper) for a module. kind is uppercase schema token."""
    hyper: Dict[str, Any] = {}
    name = _cls_name(m)

    # Containers first (before leaf isinstance that might not apply)
    if isinstance(m, nn.Sequential):
        return "SEQUENTIAL", hyper
    if isinstance(m, nn.ModuleList):
        return "MODULE_LIST", hyper
    if isinstance(m, nn.ModuleDict):
        return "MODULE_DICT", hyper

    # Parametric leaves
    if isinstance(m, nn.Linear):
        hyper = {
            "in_features": int(m.in_features),
            "out_features": int(m.out_features),
            "bias": m.bias is not None,
        }
        return "LINEAR", hyper
    if isinstance(m, nn.Embedding):
        hyper = {
            "num_embeddings": int(m.num_embeddings),
            "embedding_dim": int(m.embedding_dim),
            "padding_idx": int(m.padding_idx) if m.padding_idx is not None else None,
        }
        return "EMBEDDING", hyper
    if isinstance(m, nn.LayerNorm):
        ns = m.normalized_shape
        if isinstance(ns, (tuple, list)):
            ns_list = [int(x) for x in ns]
        else:
            ns_list = [int(ns)]
        hyper = {
            "normalized_shape": ns_list,
            "eps": float(m.eps),
            "elementwise_affine": bool(m.elementwise_affine),
        }
        return "LAYER_NORM", hyper
    if isinstance(m, nn.BatchNorm1d):
        hyper = {
            "num_features": int(m.num_features),
            "eps": float(m.eps),
            "momentum": float(m.momentum) if m.momentum is not None else None,
            "affine": bool(m.affine),
            "track_running_stats": bool(m.track_running_stats),
            "dim": 1,
        }
        return "BATCH_NORM_1D", hyper
    if isinstance(m, nn.BatchNorm2d):
        hyper = {
            "num_features": int(m.num_features),
            "eps": float(m.eps),
            "momentum": float(m.momentum) if m.momentum is not None else None,
            "affine": bool(m.affine),
            "track_running_stats": bool(m.track_running_stats),
            "dim": 2,
        }
        return "BATCH_NORM_2D", hyper
    if isinstance(m, nn.BatchNorm3d):
        hyper = {
            "num_features": int(m.num_features),
            "eps": float(m.eps),
            "momentum": float(m.momentum) if m.momentum is not None else None,
            "affine": bool(m.affine),
            "track_running_stats": bool(m.track_running_stats),
            "dim": 3,
        }
        return "BATCH_NORM_3D", hyper
    if isinstance(m, nn.GroupNorm):
        hyper = {
            "num_groups": int(m.num_groups),
            "num_channels": int(m.num_channels),
            "eps": float(m.eps),
            "affine": bool(m.affine),
        }
        return "GROUP_NORM", hyper
    if isinstance(m, nn.Conv1d):
        hyper = {
            "in_channels": int(m.in_channels),
            "out_channels": int(m.out_channels),
            "kernel_size": _as_int_tuple(m.kernel_size, 1),
            "stride": _as_int_tuple(m.stride, 1),
            "padding": _as_int_or_tuple(m.padding),
            "dilation": _as_int_tuple(m.dilation, 1),
            "groups": int(m.groups),
            "bias": m.bias is not None,
        }
        return "CONV_1D", hyper
    if isinstance(m, nn.Conv2d):
        hyper = {
            "in_channels": int(m.in_channels),
            "out_channels": int(m.out_channels),
            "kernel_size": _as_int_tuple(m.kernel_size, 2),
            "stride": _as_int_tuple(m.stride, 2),
            "padding": _as_int_or_tuple(m.padding),
            "dilation": _as_int_tuple(m.dilation, 2),
            "groups": int(m.groups),
            "bias": m.bias is not None,
        }
        return "CONV_2D", hyper
    if isinstance(m, nn.Conv3d):
        hyper = {
            "in_channels": int(m.in_channels),
            "out_channels": int(m.out_channels),
            "kernel_size": _as_int_tuple(m.kernel_size, 3),
            "stride": _as_int_tuple(m.stride, 3),
            "padding": _as_int_or_tuple(m.padding),
            "dilation": _as_int_tuple(m.dilation, 3),
            "groups": int(m.groups),
            "bias": m.bias is not None,
        }
        return "CONV_3D", hyper

    # Param-free activations / util
    if isinstance(m, nn.ReLU):
        hyper = {"inplace": bool(m.inplace)}
        return "RELU", hyper
    if isinstance(m, nn.ReLU6):
        hyper = {"inplace": bool(m.inplace)}
        return "RELU6", hyper
    if isinstance(m, nn.LeakyReLU):
        hyper = {"negative_slope": float(m.negative_slope), "inplace": bool(m.inplace)}
        return "LEAKY_RELU", hyper
    if isinstance(m, nn.GELU):
        # approximate may exist on newer torch
        approx = getattr(m, "approximate", "none")
        hyper = {"approximate": str(approx)}
        return "GELU", hyper
    if isinstance(m, nn.SiLU):
        hyper = {"inplace": bool(getattr(m, "inplace", False))}
        return "SILU", hyper
    if isinstance(m, nn.Tanh):
        return "TANH", hyper
    if isinstance(m, nn.Sigmoid):
        return "SIGMOID", hyper
    if isinstance(m, nn.Softmax):
        hyper = {"dim": int(m.dim) if m.dim is not None else -1}
        return "SOFTMAX", hyper
    if isinstance(m, nn.LogSoftmax):
        hyper = {"dim": int(m.dim) if m.dim is not None else -1}
        return "LOG_SOFTMAX", hyper
    if isinstance(m, nn.Dropout):
        hyper = {"p": float(m.p), "inplace": bool(m.inplace)}
        return "DROPOUT", hyper
    if isinstance(m, nn.Identity):
        return "IDENTITY", hyper
    if isinstance(m, nn.Flatten):
        hyper = {"start_dim": int(m.start_dim), "end_dim": int(m.end_dim)}
        return "FLATTEN", hyper

    # FuxiCTR / custom: Dice often wraps BN — treat as COMPOSITE with children
    # (class_name preserved so Java can still nest BN).
    # Bare Parameter modules don't exist as nn.Module usually.

    # Unknown with children → COMPOSITE; leaf unknown → COMPOSITE empty
    return "COMPOSITE", hyper


def _as_int_tuple(v, n: int) -> List[int]:
    if isinstance(v, (tuple, list)):
        return [int(x) for x in v]
    return [int(v)] * n


def _as_int_or_tuple(v) -> Any:
    if isinstance(v, (tuple, list)):
        return [int(x) for x in v]
    return int(v)


# ---------------------------------------------------------------------------
# Walk
# ---------------------------------------------------------------------------

def build_structure(model: nn.Module, root_name: Optional[str] = None) -> Dict[str, Any]:
    """Build schema-v2 dict from a live Module."""
    if root_name is None:
        root_name = _cls_name(model)

    nodes: Dict[str, Any] = {}

    def walk(m: nn.Module, path: str) -> None:
        kind, hyper = _classify(m)
        children_names: List[str] = []
        # Preserve registration order
        for name, child in m.named_children():
            children_names.append(name)
        node: Dict[str, Any] = {
            "kind": kind,
            "class_name": _cls_name(m),
        }
        if children_names:
            node["children"] = children_names
        if hyper:
            node["hyper"] = hyper
        # Direct parameters registered on this module (not children)
        own_params = [n for n, _ in m.named_parameters(recurse=False)]
        own_bufs = [n for n, _ in m.named_buffers(recurse=False)]
        if own_params:
            node["own_parameters"] = own_params
        if own_bufs:
            node["own_buffers"] = own_bufs
        nodes[path] = node

        for name, child in m.named_children():
            child_path = name if path == "" else f"{path}.{name}"
            walk(child, child_path)

        # Bare nn.Parameter attributes that are also in _parameters already
        # covered by own_parameters; StructureModuleBuilder uses state_dict keys.

    walk(model, "")

    # Full state_dict key lists
    parameters = list(model.state_dict().keys())
    # buffers only (state_dict includes both; separate for clarity)
    buf_keys = []
    param_set = {n for n, _ in model.named_parameters()}
    for k in parameters:
        # buffer keys typically running_* / num_batches_tracked or not in named_parameters
        leaf = k.rsplit(".", 1)[-1]
        if k not in param_set or leaf in (
            "running_mean", "running_var", "num_batches_tracked",
        ):
            # still list all; Java uses parameters list as bind order
            pass
    for n, _ in model.named_buffers():
        buf_keys.append(n)

    return {
        "version": 2,
        "root": root_name,
        "nodes": nodes,
        "parameters": parameters,
        "buffers": buf_keys,
    }


def dump_structure(model: nn.Module, out_path: str | Path,
                   root_name: Optional[str] = None) -> Dict[str, Any]:
    """Write structure JSON and return the dict."""
    out_path = Path(out_path)
    data = build_structure(model, root_name=root_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return data


def structure_to_compact_meta(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Compact path→kind token map compatible with existing encodeStructureMeta
    (e.g. mlp.2=DROPOUT:0.1). Useful for embedding in safetensors metadata.
    """
    out: Dict[str, str] = {}
    for path, node in data.get("nodes", {}).items():
        kind = node.get("kind", "COMPOSITE")
        hyper = node.get("hyper") or {}
        token = kind
        if kind == "DROPOUT" and "p" in hyper:
            token = f"DROPOUT:{hyper['p']}"
        elif kind == "SOFTMAX" and "dim" in hyper:
            token = f"SOFTMAX:{hyper['dim']}"
        elif kind == "LEAKY_RELU" and "negative_slope" in hyper:
            token = f"LEAKY_RELU:{hyper['negative_slope']}"
        elif kind == "COMPOSITE":
            cn = node.get("class_name") or "COMPOSITE"
            token = f"COMPOSITE:{cn}"
        if path == "":
            continue  # root optional in compact meta
        out[path] = token
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output .structure.json path")
    ap.add_argument("--root-name", default=None, help="override root class name")
    ap.add_argument(
        "--demo",
        action="store_true",
        help="dump a tiny demo Sequential (Linear-ReLU-Dropout-Linear-Softmax)",
    )
    ap.add_argument(
        "--from-state-print",
        default=None,
        help="(debug) only write empty shell — not for production",
    )
    args = ap.parse_args(argv)

    if args.demo:
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 4),
            nn.Softmax(dim=-1),
        )
        dump_structure(model, args.out, root_name=args.root_name or "DemoMLP")
        print(f"wrote demo structure → {args.out}")
        return 0

    print(
        "dump_module_structure.py is primarily a library.\n"
        "Call dump_structure(model, path) from your training script,\n"
        "or use --demo to emit a sample structure file.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
