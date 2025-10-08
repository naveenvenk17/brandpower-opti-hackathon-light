#!/usr/bin/env python3
"""
BrandCompass - Main Entry Point
Single unified FastAPI application serving both frontend and backend

Usage:
    python main.py              # Run server (default port 8010)
    python main.py --port 5000  # Run on custom port
"""
import sys
import argparse
import uvicorn

def main():
    parser = argparse.ArgumentParser(
        description='BrandCompass - Brand Power Optimization Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start on port 8010 (default)
  python main.py --port 5000        # Start on port 5000
  python main.py --host 127.0.0.1   # Bind to localhost only
        """
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8010,
        help='Port to run the server on (default: 8010)'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        default=True,
        help='Enable auto-reload on code changes (default: True)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("  BrandCompass.ai - Brand Power Optimization Platform")
    print("="*70)
    print()
    print("🚀 Starting unified FastAPI application...")
    print(f"📍 Web Interface:  http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
    print(f"📖 API Docs:       http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/docs")
    print(f"🔧 Interactive:    http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}/redoc")
    print()
    print("Features:")
    print("  ✓ Frontend UI (HTML pages)")
    print("  ✓ REST API endpoints")
    print("  ✓ Real-time forecasting")
    print("  ✓ Marketing optimization")
    print("  ✓ AI agent chat")
    print()
    print("Press CTRL+C to stop the server")
    print("="*70)
    print()
    
    # Change working directory to parent if needed
    import os
    if os.path.basename(os.getcwd()) == 'src':
        os.chdir('..')
    
    uvicorn.run(
        "src.app:app",
        host=args.host,
        port=args.port,
        log_level="info",
        reload=args.reload
    )

if __name__ == '__main__':
    main()

