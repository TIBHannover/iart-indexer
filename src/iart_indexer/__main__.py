import sys

from .indexer_main import main

# handling segmentation fault
import faulthandler

faulthandler.enable()
PYTHONFAULTHANDLER = 1

if __name__ == "__main__":
    sys.exit(main())
