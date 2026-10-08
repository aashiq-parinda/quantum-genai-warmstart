#!/usr/bin/env python3
"""
Quantum GenAI Warm-Start Lab — Local HTTP Server & API Bridge
Zero-dependency Python 3 server to run and visualize the research locally.
"""

import os
import sys
import json
import socket
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT_PREFERRED = 8000
DIR_VISUALIZER = os.path.dirname(os.path.abspath(__file__))

# Check if qwarmstart is importable in the current environment
HAS_QWARMSTART = False
try:
    import qwarmstart
    HAS_QWARMSTART = True
except ImportError:
    pass

class WarmStartRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler with API endpoints and static file serving."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR_VISUALIZER, **kwargs)

    def do_GET(self):
        # API: Status & Diagnostics
        if self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = {
                "status": "ok",
                "qwarmstart_available": HAS_QWARMSTART,
                "python_version": sys.version.split()[0],
                "port": self.server.server_port,
            }
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        # API: Precomputed benchmark JSON
        if self.path == "/api/benchmark":
            benchmark_path = os.path.join(DIR_VISUALIZER, "benchmark_data.json")
            if os.path.exists(benchmark_path):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                with open(benchmark_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        # Normal static file serving
        return super().do_GET()

    def log_message(self, format, *args):
        # Cleaner console logs
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


def find_free_port(start_port=PORT_PREFERRED):
    """Finds an open port starting from start_port."""
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def main():
    port = find_free_port()
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, WarmStartRequestHandler)

    url = f"http://localhost:{port}"

    print("=" * 72)
    print(" ⚛️  QUANTUM GENAI WARM-START LAB — LOCAL VISUALIZER")
    print("=" * 72)
    print(f" 🚀 Server running at: \033[1;36m{url}\033[0m")
    print(f" 📂 Serving directory: {DIR_VISUALIZER}")
    print(f" 🧬 Python Package (qwarmstart): {'✅ Detected' if HAS_QWARMSTART else '⚡ Standalone Client Mode'}")
    print("=" * 72)
    print(" Press Ctrl+C anytime to stop the server.\n")

    # Automatically launch web browser
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n Shutting down Quantum Warm-Start Lab server.")
        httpd.server_close()


if __name__ == "__main__":
    main()
