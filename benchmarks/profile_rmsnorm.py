"""Nsight Systems entry point for the repository's RMSNorm CUDA workload."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy
import torch
import os



REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from torch_llm.model.rmsnorm import RMSNorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    print("PROFILE PYTHON PID:", os.getpid())
    args = parse_args()
    print(f"python={sys.executable}")
    print(f"torch={torch.__version__} torch.version.cuda={torch.version.cuda}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in this interpreter. Install a CUDA-enabled PyTorch "
            "wheel and run this script with that same interpreter."
        )

    device = torch.device("cuda:0")
    print(
        f"device={torch.cuda.get_device_name(device)} "
        f"capability={torch.cuda.get_device_capability(device)}"
    )

    torch.manual_seed(0)
    x = torch.randn(
        args.batch_size,
        args.tokens,
        args.d_model,
        device=device,
        dtype=torch.float32,
    )
    rms_norm = RMSNorm(args.d_model, dtype=x.dtype).to(device).eval()

    with torch.inference_mode():
        for _ in range(args.warmup):
            output = rms_norm(x)
        torch.cuda.synchronize(device)

        torch.cuda.nvtx.range_push("rmsnorm_cuda_kernels")
        for _ in range(args.iterations):
            output = rms_norm(x)
        torch.cuda.synchronize(device)
        torch.cuda.nvtx.range_pop()

    print(f"completed_cuda_iterations={args.iterations}")
    print(f"output_device={output.device} output_mean={output.mean().item():.6f}")


if __name__ == "__main__":
    main()



"""nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true --output=profiles/rmsnorm_cuda .\.venv\Scripts\python.exe benchmarks/profile_rmsnorm.py; if ($LASTEXITCODE -eq 0) { nsys stats --force-export=true --report cuda_gpu_kern_sum profiles/rmsnorm_cuda.nsys-rep }"""