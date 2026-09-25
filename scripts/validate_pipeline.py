"""Print a validation table for inputs and every generated pipeline artifact."""

import argparse

from src.utils.pipeline_validation import validation_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument(
        "--fast", action="store_true", help="Skip counting TFRecord entries"
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return success even when later pipeline stages have not been run",
    )
    args = parser.parse_args()
    report = validation_frame(args.config, include_record_counts=not args.fast)
    print(report.to_string(index=False))
    failures = report[report.status == "FAIL"]
    if len(failures) and not args.allow_incomplete:
        raise SystemExit(f"\n{len(failures)} validation check(s) failed")


if __name__ == "__main__":
    main()
