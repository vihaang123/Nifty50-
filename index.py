"""Vercel entry point for the FastAPI backend. Local development still uses: uvicorn api.main:app --reload"""
import os
import tempfile

# Vercel's filesystem is read-only apart from the system temp folder; matplotlib wants a writable config/cache folder.
# tempfile finds that folder on any machine, so no absolute path is baked in.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

from api.main import app  # noqa: E402,F401
