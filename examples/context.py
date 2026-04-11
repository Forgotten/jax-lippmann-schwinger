# This is small code to run the example files in this repo without
# installing the package in your environment.

import os
import sys

import jax_ls

sys.path.insert(
  0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
