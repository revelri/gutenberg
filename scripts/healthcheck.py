#!/usr/bin/env python3
"""Health check script for all Gutenberg services."""

import sys

import httpx

SERVICES = {
    "Ollama": "http://localhost:11434/api/version",
    "ChromaDB": "http://localhost:8200/api/v1/heartbeat",
    "Gutenberg API": "http://localhost:8002/api/health",
    "LibreChat": "http://localhost:3080/api/health",
}


def main():
    all_ok = True
    for name, url in SERVICES.items():
        try:
            r = httpx.get(url, timeout=5)
            if r.status_code == 200:
                print(f"  {name}: OK")
            else:
                print(f"  {name}: HTTP {r.status_code}")
                all_ok = False
        except Exception as e:
            print(f"  {name}: UNREACHABLE ({e})")
            all_ok = False

    print()
    if all_ok:
        print("All services healthy.")
    else:
        print("Some services are unhealthy.")
        sys.exit(1)


if __name__ == "__main__":
    main()
