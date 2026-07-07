"""Convenience prediction entry point.

This script delegates to ``train.run_pipeline`` because the final prediction
file depends on the out-of-fold blend search described in the project report.
"""

from __future__ import annotations

from .train import main


if __name__ == "__main__":
    main()
