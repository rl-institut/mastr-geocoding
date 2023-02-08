import sys

from loguru import logger

from mastr_geocoding import run_mastr_geocoding

logger.remove()
logger.add(sys.stderr, level="DEBUG")

if __name__ == "__main__":
    run_mastr_geocoding()
