"""Run mlx_lm_lora.train with MLX memory capped, so a too-large step stops the job instead of macOS.

A GRPO run kernel-panicked the 48 GB M5 Pro (watchdog timeout, 21 swap files). mlx-lm's BatchGenerator
raises the GPU wired limit to the recommended 37.4 GB at every rollout, MLX's own memory limit
defaults to 45.6 GB, and batches with the longest prompts pushed macOS into swap until it froze.

  soft limit  mx.set_memory_limit, and the wired limit (replacing mlx-lm's 37.4 GB). MLX frees its
              cache and waits for pending work before growing past it, but it is advisory: MLX can
              still allocate beyond it.
  hard limit  a watchdog thread exits the process (code 3) when MLX's active memory passes it, before
              the OS starts thrashing.

train_grpo.py runs `python src/mlx_guard.py <mlx_lm_lora.train args>` with MLX_SOFT_LIMIT_GB and
MLX_HARD_LIMIT_GB set.
"""

import importlib
import os
import runpy
import sys
import threading
import time

import mlx.core as mx

GB = 2**30


def install(soft_gb, hard_gb, cache_gb=2.0, poll_s=0.2):
    soft = int(soft_gb * GB)
    mx.set_memory_limit(soft)
    mx.set_cache_limit(int(cache_gb * GB))
    wired = min(soft, mx.device_info()["max_recommended_working_set_size"])
    mx.set_wired_limit(wired)
    # BatchGenerator looks this up in mlx_lm.generate's globals on every rollout. Import the module
    # via importlib: the attribute mlx_lm.generate is shadowed by the generate() function.
    generate_module = importlib.import_module("mlx_lm.generate")
    generate_module.maybe_set_recommended_wired_limit = lambda: mx.set_wired_limit(wired)

    def watch():
        while True:
            used = mx.get_active_memory()
            if used > hard_gb * GB:
                sys.stderr.write(f"\n[mlx_guard] MLX active memory {used / GB:.1f} GB passed the {hard_gb:g} GB hard "
                                 f"limit; stopping before macOS starts swapping. Lower --micro-batch-size or "
                                 f"--max-completion-length, or raise --gpu-kill-gb if other apps are closed.\n")
                sys.stderr.flush()
                os._exit(3)
            time.sleep(poll_s)

    threading.Thread(target=watch, daemon=True).start()


if __name__ == "__main__":
    install(float(os.environ.get("MLX_SOFT_LIMIT_GB", 28)), float(os.environ.get("MLX_HARD_LIMIT_GB", 34)))
    sys.argv = ["mlx_lm_lora.train", *sys.argv[1:]]
    runpy.run_module("mlx_lm_lora.train", run_name="__main__")
