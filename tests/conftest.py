import os
import sys

import matplotlib

# Tests import the project modules from the repository root and never open plot windows
matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
