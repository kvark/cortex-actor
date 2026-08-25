"""Export a training checkpoint as a sanitized release artifact.

Keeps only what deployment needs: model weights, the architecture/contract
arguments, and the action schema. Drops optimizer, scheduler, RNG, sampler
cursor, and training history, and scrubs local filesystem paths from the
stored arguments.

Usage:
    python scripts/export_checkpoint.py <in.pt> <out.pt> [--fp16]
"""

import argparse

import torch

PATH_ARG_KEYS = (
    "feat_dir",
    "pose_dir",
    "raw_dir",
    "out",
    "output_dir",
    "corpus_index",
    "packed_corpus",
    "resume",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--fp16", action="store_true", help="store weights as float16")
    args = parser.parse_args()

    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    model = ck["model"]
    if args.fp16:
        model = {
            k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in model.items()
        }
    args_dict = dict(ck.get("args_dict") or {})
    for key in list(args_dict):
        if key in PATH_ARG_KEYS or (
            isinstance(args_dict[key], str) and args_dict[key].startswith("/")
        ):
            args_dict.pop(key)

    out = {
        "schema_version": ck.get("schema_version"),
        "model": model,
        "args_dict": args_dict,
        "action_schema": ck.get("action_schema"),
        "step": ck.get("step"),
    }
    torch.save(out, args.dst)
    size = sum(v.numel() * v.element_size() for v in model.values() if torch.is_tensor(v))
    print(f"wrote {args.dst}: {len(model)} tensors, {size / 1e6:.1f} MB of weights")


if __name__ == "__main__":
    main()
