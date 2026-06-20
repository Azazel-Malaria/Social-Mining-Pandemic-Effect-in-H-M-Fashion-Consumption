from __future__ import annotations

# Backward-compatible entry point.  The cleaned pipeline now calls
# util.social_inference directly, but older scripts may still use this module.
from util.social_inference import main

if __name__ == '__main__':
    main()
