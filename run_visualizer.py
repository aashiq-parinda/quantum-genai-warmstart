#!/usr/bin/env python3
"""Convenient runner for the Quantum GenAI Warm-Start Visualizer."""
import sys
import os

# Redirect execution into visualizer/server.py
current_dir = os.path.dirname(os.path.abspath(__file__))
server_script = os.path.join(current_dir, "visualizer", "server.py")
os.execv(sys.executable, [sys.executable, server_script] + sys.argv[1:])
