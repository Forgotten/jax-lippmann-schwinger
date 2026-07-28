# This is small code to run the test files in this repo without
# installing the package in your environment.
#
# The path has to be extended *before* the package is imported.

import os
import sys

sys.path.insert(
  0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

import jax_ls  # noqa: E402,F401
