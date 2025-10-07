"""
Main entry point for CLI execution

Usage:
    python -m production_scripts forecast --country X --brand Y
    python -m production_scripts simulate --country X --brand Y --quarter Qtr3 --allocation "channel:amount"
    python -m production_scripts list forecasters
"""

from production_scripts.cli import main

if __name__ == '__main__':
    exit(main())
