import os
import sys

# Add project root to path so 'engine' can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Set Numba cache to /tmp for read-only environments (Vercel)
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"

from engine.app import app
