#!/usr/bin/env python3
"""
Simple script to run the Flask application
"""
import os
import sys

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['flask', 'pandas', 'numpy', 'plotly', 'openpyxl']
    missing = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print(f"\nPlease install them using:")
        print(f"    pip install {' '.join(missing)}")
        print(f"\nOr install all requirements:")
        print(f"    pip install -r requirements_flask.txt")
        return False

    return True


def main():
    print("=" * 60)
    print("BrandCompass.ai - Flask Web Application")
    print("=" * 60)
    print()

    if not check_dependencies():
        sys.exit(1)

    print("✓ All dependencies are installed")
    print()
    print("Starting Flask application...")
    print()
    print("Access the application at: http://localhost:5000")
    print("Press CTRL+C to stop the server")
    print()
    print("=" * 60)
    print()

    from app import app
    app.run(debug=True, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
