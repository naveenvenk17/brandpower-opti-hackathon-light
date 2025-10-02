#!/usr/bin/env python3
"""
Simple script to run the BrandCompass.ai Streamlit application
"""

import subprocess
import sys
import os


def main():
    # Change to frontend directory
    frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')
    os.chdir(frontend_dir)

    # Run streamlit app
    try:
        subprocess.run([sys.executable, '-m', 'streamlit',
                       'run', 'main.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
