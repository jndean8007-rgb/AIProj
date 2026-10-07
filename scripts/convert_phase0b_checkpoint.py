"""One-time conversion of checkpoints saved before Phase 0b step 2 to the new parameter names (handoff D17).

Step 2 renamed the model's attributes, which renames every checkpoint key under them:
  blocks.{i}.attention / attn_norm / moe / moe_norm  ->  blocks.{i}.mixer / mixer_norm / ffn / ffn_norm
  final_norm  ->  head.norm
  lm_head     ->  head.proj
Only that one attribute part of each key is renamed; everything else (embedding.weight,
q_proj.weight, router.ema, ...) passes through unchanged. Tensors are not modified.

Usage (from the repository root):
    python scripts/convert_phase0b_checkpoint.py PATH [PATH ...]

Each file is first copied to PATH.pre_step2.bak, then overwritten in place. Keeping the same
path means anything that records it (the golden test, path_config) keeps working. Training
checkpoints ({"model": ..., "AdamW": ..., ...}) and plain state dicts are both handled; optimizer
state is stored by parameter position, not name, so it needs no change. Files already in the
new format are left untouched.

Delete this script once every checkpoint you still need has been converted.
"""

import shutil
import sys
from pathlib import Path

import torch as t

BLOCK_RENAMES = {"attention": "mixer", "attn_norm": "mixer_norm", "moe": "ffn", "moe_norm": "ffn_norm"}
TOP_LEVEL_RENAMES = {"final_norm": "head.norm", "lm_head": "head.proj"}


def convert_key(key: str) -> str:
    parts = key.split(".")
    if len(parts) > 2 and parts[0] == "blocks" and parts[2] in BLOCK_RENAMES:
        parts[2] = BLOCK_RENAMES[parts[2]]
    elif parts[0] in TOP_LEVEL_RENAMES:
        parts[0] = TOP_LEVEL_RENAMES[parts[0]]
    return ".".join(parts)


def convert_state_dict(state_dict: dict) -> dict:
    """Same tensors, same order, new names. No-op on new-format dicts."""
    converted = {convert_key(key): value for key, value in state_dict.items()}
    if len(converted) != len(state_dict):
        raise ValueError("two keys converted to the same name; refusing to drop a tensor")
    return converted


def convert_file(path: Path) -> None:
    checkpoint = t.load(path, map_location="cpu", weights_only=True)
    is_training_checkpoint = "model" in checkpoint
    state_dict = checkpoint["model"] if is_training_checkpoint else checkpoint

    converted = convert_state_dict(state_dict)
    if list(converted) == list(state_dict):
        print(f"{path}: already in the new format, left unchanged")
        return

    backup = path.with_name(path.name + ".pre_step2.bak")
    if backup.exists():
        raise FileExistsError(f"{backup} already exists; not overwriting an existing backup")
    shutil.copy2(path, backup)

    t.save({**checkpoint, "model": converted} if is_training_checkpoint else converted, path)
    print(f"{path}: converted ({sum(a != b for a, b in zip(state_dict, converted))} keys renamed); backup at {backup}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for argument in sys.argv[1:]:
        convert_file(Path(argument))
