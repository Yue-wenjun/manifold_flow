"""
web/websocket.py

WebSocket streaming manager for Manifold Flow v2.0.
Executes N-dimensional math in background threads and streams only 3D projections to clients.
"""

import time
import numpy as np
import threading
from typing import Dict, Any
from flask_socketio import SocketIO, emit

from ..core.base_system import DeterministicSystem, StochasticSystem


def _rk4_step(system: DeterministicSystem, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    k1 = system.drift(t, y)
    k2 = system.drift(t + dt/2, y + dt * k1 / 2)
    k3 = system.drift(t + dt/2, y + dt * k2 / 2)
    k4 = system.drift(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _euler_maruyama_step(system: StochasticSystem, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    f = system.drift(t, y)
    g = system.diffusion(t, y)
    dW = np.random.normal(0, 1.0, size=system.state_dim) * np.sqrt(dt)
    return y + f * dt + g * dW

class WebSocketManager:
    def __init__(self, app):
        self.socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self._register_handlers()

    def _register_handlers(self):
        @self.socketio.on('connect')
        def handle_connect():
            print("Client connected")
            emit('status', {'message': 'Connected to Manifold Flow v2.0 Engine'})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print("Client disconnected")
            pass

        @self.socketio.on('start_stream')
        def handle_start_stream(data):
            stream_id = data.get('stream_id')
            system_name = data.get('system_id')

            if not stream_id or not system_name:
                emit('stream_error', {'message': 'Missing stream_id or system_id'})
                return

            self.start_trajectory_stream(stream_id, data)

        @self.socketio.on('stop_stream')
        def handle_stop_stream(data):
            stream_id = data.get('stream_id')
            if stream_id in self.active_streams:
                self.active_streams[stream_id]['active'] = False
                emit('status', {'message': f'Stream {stream_id} stopped'})

        @self.socketio.on('update_parameters')
        def handle_update_parameters(data):
            stream_id = data.get('stream_id')
            new_params = data.get('parameters', {})

            if stream_id in self.active_streams:
                system = self.active_streams[stream_id]['system']
                system.update_parameters(new_params)
                emit('status', {'message': f'Parameters updated for {stream_id}', 'current_params': system.parameters})

    def start_trajectory_stream(self, stream_id: str, config: dict):
        if stream_id in self.active_streams:
            self.active_streams[stream_id]['active'] = False
            time.sleep(0.1)

        system_name = config.get('system_id')
        params_override = config.get('parameters', {})

        try:
            from ..systems.registry import get_system

            system = get_system(system_name, **params_override)
            state = system.get_initial_conditions()

            self.active_streams[stream_id] = {
                'active': True,
                'system': system,
                'state': state,
                'time': 0.0,
                'thread': None
            }

            thread = threading.Thread(
                target=self._trajectory_streaming_worker,
                args=(stream_id, config)
            )
            thread.daemon = True
            self.active_streams[stream_id]['thread'] = thread
            thread.start()

            emit('stream_started', {'stream_id': stream_id, 'system': system_name})

        except Exception as e:
            emit('stream_error', {'message': str(e)})

    def _trajectory_streaming_worker(self, stream_id: str, config: dict):
        stream_data = self.active_streams.get(stream_id)
        if not stream_data:
            return

        system = stream_data['system']

        dt = config.get('time_step', 0.01)
        steps_per_emit = config.get('steps_per_emit', 5)
        emit_interval = config.get('update_interval', 0.033)  # ~30 FPS

        is_sde = isinstance(system, StochasticSystem)

        while self.active_streams.get(stream_id, {}).get('active', False):
            loop_start = time.time()

            y = self.active_streams[stream_id]['state']
            t = self.active_streams[stream_id]['time']

            for _ in range(steps_per_emit):
                if is_sde:
                    y = _euler_maruyama_step(system, t, y, dt)
                else:
                    y = _rk4_step(system, t, y, dt)
                t += dt

            self.active_streams[stream_id]['state'] = y
            self.active_streams[stream_id]['time'] = t

            proj_3d = system.project_to_3d(y)
            payload = proj_3d.tolist()

            self.socketio.emit('trajectory_update', {
                'stream_id': stream_id,
                'time': t,
                'point': payload,
                'is_scatter': len(proj_3d.shape) == 2
            })

            compute_time = time.time() - loop_start
            sleep_time = max(0, emit_interval - compute_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

def create_websocket_app(app) -> WebSocketManager:
    return WebSocketManager(app)
