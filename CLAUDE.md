# Python enviromnent

Use `uv` alongside the virtual environment when running Python scripts or notebooks.

When requiring environment variables, use:

```bash
uv run --env-file keys.env main.py
```

Do not edit the `pyproject.toml` file manually. Always use `uv add` to let uv add Python packages to the virtual environment.