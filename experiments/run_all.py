"""
experiments/run_all.py

Master runner — executes all experiments in order and saves every figure
to manifold_flow2.0/figures/.

Run from anywhere:
    python experiments/run_all.py
    # or from inside experiments/:
    python run_all.py
"""

import sys
import os
import traceback

# Ensure manifold_flow package is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

EXPERIMENTS = [
    ("CANDY ablations (exp1-5)",     "candy_diffusion",             "main"),
    ("Diffusion ablations (diff_exp1-4)", "run_diffusion",          "main"),
    ("Neural (Hopfield + Transformer)",  "neural_experiment",       "main"),
    ("NeuralODE vs U-Net dynamics",      "ode_unet_experiment",     "main"),
    ("CANDY convergence",                "candy_experiment",        "main"),
    ("Diffusion systems",                "diffusion_experiment",    "main"),
    ("Manifold learning (t-SNE/UMAP)",   "manifold_experiment",    "main"),
    ("Hopfield multimodal (Bunny)",      "neural_multimodal_experiment",      "main"),
    ("NeuralODE+UNet multimodal",        "ode_unet_multimodal_experiment",    "main"),
    ("CANDY multimodal (Bunny)",         "candy_multimodal_experiment",       "main"),
    ("Diffusion multimodal",             "diffusion_multimodal_experiment",   "main"),
    ("Manifold multimodal",              "manifold_multimodal_experiment",    "main"),
]


def run_all():
    print("=" * 60)
    print("Running all experiments -> figures/")
    print(f"Output: {os.path.abspath(FIGURES_DIR)}")
    print("=" * 60)

    passed, failed = [], []

    for label, module_name, func_name in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"[START] {label}")
        print(f"{'='*60}")
        try:
            import importlib
            mod = importlib.import_module(module_name)
            importlib.reload(mod)          # re-import in case of shared state
            fn = getattr(mod, func_name)
            fn()
            passed.append(label)
            print(f"[DONE]  {label}")
        except Exception as e:
            failed.append((label, e))
            print(f"[FAIL]  {label}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("Failed experiments:")
        for label, err in failed:
            print(f"  - {label}: {err}")
    print("=" * 60)
    print(f"\nAll figures saved to: {os.path.abspath(FIGURES_DIR)}")
    print("Upload the figures/ folder to Overleaf alongside reports.tex")


if __name__ == "__main__":
    # Must run from the experiments/ directory so relative imports resolve
    experiments_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(experiments_dir)
    run_all()
