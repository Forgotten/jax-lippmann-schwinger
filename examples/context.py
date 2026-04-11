# this is small code to run the example files in this repo without
# installing the package in your environment

import jax_ls
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
