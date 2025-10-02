#!/usr/bin/env python3
"""
Run script for BrandCompass.ai Streamlit application
"""

import subprocess
import sys
import os


def main():
    """Run the BrandCompass.ai application"""

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the main app file
    app_file = os.path.join(script_dir, "BrandCompass.py")

    # Run streamlit app
    try:
        print("🚀 Starting BrandCompass.ai...")
        print("📍 Navigate to the URL shown below to access the application")
        print("🔄 Press Ctrl+C to stop the application")
        print("-" * 50)

        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', app_file,
            '--server.port', '8505',
            '--server.headless', 'false'
        ], check=True)

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running BrandCompass.ai: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
