# Project: ERBS (Enhanced Representation-Based Sampling)

## Coding Standards
- **Framework:** We use JAX for computing operations and ASE as a molecular dynamics engine.
- **Functional Programming:** prefer pure functions. Avoid side effects.
- **Arrays:** JAX arrays are immutable. Never use `x[i] = y`. Use `x = x.at[i].set(y)`.
- **Style:** Follow PEP 8 but allow for JAX-specific conventions (e.g., `def fn(x, y)` is fine for vmapping).
- **Project** The project is configured with `uv`. You therefore have to use `uv run` before running python scripts.

## Common Pitfalls to Avoid
- Watch out for 64-bit precision; we default to float32 for performance unless specified. Especially positions, cells and the final predictions need to be in float 64.
- Remember that `jax.jit` cannot handle dynamic shapes.

## Testing
- Run tests using `uv run coverage run -m pytest -k "not slow"`.


## 4. Documentation Style
- Use Numppy-style docstrings.
- Include shapes in docstrings: `x: Array[float32, "batch atoms 3"]`.
