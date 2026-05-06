# Manifold Flow v2.0

A real-time 3D visualization engine for dynamical systems — classical attractors, neural network dynamics, diffusion models, and manifold learning — rendered as interacting particle swarms in the browser.

**Live demo:** [manifold-flow.onrender.com/visualization/hybrid](https://manifold-flow.onrender.com/visualization/hybrid)

> The live demo runs on a free-tier server. If the page takes ~30 seconds to load on first visit, it is waking up from sleep.

---

## Using the Interface

### 1. Select a System

Use the **Target System** dropdown to choose one of 19 dynamical systems across five categories:

| Category | Systems |
|----------|---------|
| **Classical** | Lorenz, Rössler, Chua's Circuit |
| **Shape** | Torus, Ring, Point, Line, Discrete |
| **Diffusion** | Forward Diffusion SDE, Reverse Diffusion SDE, Probability Flow ODE |
| **Manifold** | t-SNE Dynamics, UMAP Dynamics |
| **Neural** | CANDY Network, Neural ODE, Hopfield Network, Transformer Attention, U-Net, CANDY Diffusion |

### 2. Tune System Parameters

After selecting a system, sliders appear for its physics parameters (e.g. `sigma`, `rho`, `beta` for Lorenz). These update the simulation **live** — no restart needed.

### 3. Set Performance

The **⚡ Performance** section controls how hard the server works per frame:

| Slider | Range | Effect |
|--------|-------|--------|
| **Particles** | 50 – 1000 | Number of simulated particles. More particles = richer visuals, higher CPU load. |
| **Compute Steps/Frame** | 2 – 10 | ODE integration steps per WebSocket frame. Higher = smoother trajectories, higher CPU load. |

**Recommended settings for the free-tier server:**
- Particles: **50 – 200**
- Compute Steps: **2 – 3**
- For **Transformer** (O(n²) attention): keep particles **≤ 150**

### 4. Render Toggles

- **Show Trajectory History** — cyan trail lines showing each particle's recent path
- **Show Live Particles** — magenta glowing point cloud at the current position

### 5. Camera Controls

| Action | Control |
|--------|---------|
| Rotate | Left-click and drag |
| Zoom | Scroll wheel |
| Pan | Right-click and drag |
| Auto-rotate | On by default; stops when you interact |

### 6. Start / Stop

Click **START SIMULATION** to begin streaming. The server computes the trajectory in real time and pushes updates to your browser via WebSocket at ~20 fps.

Click **STOP SIMULATION** to halt the stream. You can then change the system and start a new one.

---

## Local Deployment

### Requirements

- Python 3.8 or higher
- pip

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/Yue-wenjun/manifold_flow.git
cd manifold_flow
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Start the server**

```bash
python run_server.py
```

The server starts at `http://127.0.0.1:5000` by default.

**4. Open the visualization**

Navigate to:

```
http://127.0.0.1:5000/visualization/hybrid
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `5000` | Port number |
| `DEBUG` | `True` | Flask debug mode |

Example:

```bash
PORT=8080 DEBUG=False python run_server.py
```

### Running with Gunicorn (production-like)

```bash
gunicorn -w 1 --threads 20 --bind 0.0.0.0:5000 wsgi:app
```

> Use `-w 1` (single worker). Flask-SocketIO requires a message queue for multiple workers, which is not configured here.

---

## Project Structure

```
manifold_flow/
├── core/           # Base classes for dynamical systems
├── systems/        # 19 ODE/SDE system implementations
│   ├── classical.py
│   ├── diffusion.py
│   ├── manifold.py
│   ├── neural.py
│   ├── shape.py
│   └── registry.py
├── solvers/        # RK4 and Euler-Maruyama integrators
├── web/
│   ├── app.py      # Flask routes
│   └── websocket.py # Real-time streaming engine
├── static/js/      # Three.js renderer
└── templates/      # HTML pages
run_server.py       # Local entry point
wsgi.py             # Production entry point (Gunicorn / Render)
```

---

## Adding a New System

1. Create a class in the appropriate `systems/*.py` file inheriting from `DeterministicSystem` or `StochasticSystem`.
2. Implement `get_initial_conditions()`, `drift()`, and `project_to_3d()`.
3. Register it in `systems/registry.py` with `self.register(...)`.

The frontend picks it up automatically from `/api/systems`.
