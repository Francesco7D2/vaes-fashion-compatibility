"""Show which TensorFlow device this project will use."""

import argparse
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.utils.device_utilities import configure_tensorflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    args = parser.parse_args()
    try:
        report = configure_tensorflow(args.device)
    except RuntimeError as error:
        print(f"Device check failed: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
