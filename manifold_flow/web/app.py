"""
web/app.py

Flask application initialization and REST API routes for Manifold Flow v2.0.
"""

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import traceback

from ..systems.registry import list_systems, get_system_info
from .websocket import create_websocket_app

def create_app(config=None):
    """Factory function to create the Flask application."""
    app = Flask(__name__,
                static_folder='../static',
                template_folder='../templates')

    CORS(app)

    if config:
        app.config.update(config)

    register_routes(app)

    socket_manager = create_websocket_app(app)
    app.socket_manager = socket_manager

    return app

def register_routes(app: Flask):
    """Register all REST API and page routes."""

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/visualization/hybrid')
    def visualization_hybrid():
        return render_template('visualization_hybrid.html')

    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy", "version": "2.0"})

    @app.route('/api/systems', methods=['GET'])
    def get_systems():
        """Returns all registered dynamical systems (for frontend dropdown)."""
        try:
            category = request.args.get('category')
            systems = list_systems(category)

            result = []
            for sys in systems:
                result.append({
                    "id": sys.name,
                    "name": sys.name.replace('_', ' ').title(),
                    "category": sys.category.value,
                    "description": sys.description,
                    "dimension": sys.dimension
                })

            return jsonify({"success": True, "systems": result})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/systems/<system_id>', methods=['GET'])
    def get_system_details(system_id):
        """Returns parameter metadata for a specific system (for frontend sliders)."""
        try:
            info = get_system_info(system_id)
            return jsonify({
                "success": True,
                "system": {
                    "id": info.name,
                    "category": info.category.value,
                    "description": info.description,
                    "parameters": info.parameters,
                    "documentation": info.documentation
                }
            })
        except ValueError as e:
            return jsonify({"success": False, "error": str(e)}), 404
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

def run_app(host='127.0.0.1', port=5000, debug=True):
    """
    Runner function to start the server.
    Must use socketio.run (not app.run) when WebSocket is present.
    """
    app = create_app()
    print(f"[*] Starting Manifold Flow v2.0 Engine on http://{host}:{port}")
    app.socket_manager.socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
