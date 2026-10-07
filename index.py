"""Vercel entry point for the FastAPI backend. Local development still uses: uvicorn api.main:app --reload"""
import os

# Vercel's filesystem is read-only apart from /tmp; matplotlib wants a writable config/cache folder.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from api.main import app  # noqa: E402,F401
