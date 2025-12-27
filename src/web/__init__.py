"""
DataAiPrep Web Demo Module

Provides a lightweight browser-based interface for quick evaluation.

Usage:
    # Start the web server
    python -m src.web.demo
    
    # Or from command line
    dataaiprep-web --port 8000
"""

from .demo import create_app

__all__ = ['create_app']

