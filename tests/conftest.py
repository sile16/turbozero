import os

# Ensure tests run on CPU to avoid GPU OOM in CI/dev machines.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

try:
    import jax

    jax.config.update("jax_platform_name", "cpu")
except Exception:
    # If JAX isn't importable yet, env vars above are still effective.
    pass

