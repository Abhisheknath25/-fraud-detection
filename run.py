"""
run.py — Entry point for the Fraud Detection API.

Usage
-----
    python run.py              # Start the API server
    python run.py --train      # Train the model first, then start the API
    python run.py --train-only # Train the model without starting the API
"""

import argparse
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="Fraud Detection System — Credit Card Transactions"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model before starting the API server.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train the model and exit (don't start the API).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the API server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the API server (default: 8000).",
    )
    args = parser.parse_args()

    # --- Training ---
    if args.train or args.train_only:
        print("=" * 60)
        print("  🔧 TRAINING MODE")
        print("=" * 60)
        from src.train import train
        train()
        print("\n✅ Training complete.\n")
        if args.train_only:
            return

    # --- API Server ---
    import uvicorn
    print("=" * 60)
    print("  🚀 STARTING API SERVER")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Swagger docs: http://{args.host}:{args.port}/docs")
    print("=" * 60)
    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
