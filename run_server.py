"""
run_server.py

http://127.0.0.1:5000/visualization/hybrid

Entry point for starting the Manifold Flow v2.0 Web Server.
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from manifold_flow.web.app import run_app

if __name__ == '__main__':
    HOST = os.environ.get('HOST', '127.0.0.1')
    PORT = int(os.environ.get('PORT', 5000))
    DEBUG = os.environ.get('DEBUG', 'True').lower() in ('true', '1', 't')

    print("=====================================================")
    print(" Manifold Flow v2.0 - Tensor Computing Engine")
    print("=====================================================")
    
    run_app(host=HOST, port=PORT, debug=DEBUG)