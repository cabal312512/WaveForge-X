"""Command-line interface for reproducible simulation and analysis."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

from .reproducibility import configure_local_environment


def doctor() -> dict:
    """Check interpreter, imports, local paths and headless plotting."""
    root = configure_local_environment()
    import matplotlib
    dependencies = {name: importlib.util.find_spec(name) is not None for name in
                    ("numpy", "scipy", "matplotlib", "pandas", "yaml", "tqdm")}
    writable = {}
    for name in ("results", "figures", "logs", "data/generated", ".cache/tmp"):
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryFile(dir=path):
                pass
            writable[name] = True
        except OSError:
            writable[name] = False
    return {"python": sys.version, "python_supported": sys.version_info >= (3, 11),
            "executable": sys.executable, "virtual_environment": sys.prefix != sys.base_prefix,
            "project_root": str(root), "dependencies": dependencies,
            "numba_available": importlib.util.find_spec("numba") is not None,
            "matplotlib_backend": matplotlib.get_backend(), "writable_directories": writable,
            "cache_paths": {key: os.environ.get(key) for key in
                            ("PIP_CACHE_DIR", "MPLCONFIGDIR", "NUMBA_CACHE_DIR", "TMPDIR",
                             "PYTHONPYCACHEPREFIX")}}


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments. Expected input failures return status 2."""
    configure_local_environment()
    parser = argparse.ArgumentParser(prog="waveforge6g", description="Reproducible waveform research")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "sweep"):
        sub = commands.add_parser(command, help="Run a YAML experiment")
        sub.add_argument("config", type=Path)
        sub.add_argument("--no-plot", action="store_true")
    sub = commands.add_parser("plot", help="Redraw saved results")
    sub.add_argument("directory", type=Path)
    sub.add_argument("--output-dir", type=Path)
    commands.add_parser("validate", help="Run numerical sanity checks")
    commands.add_parser("doctor", help="Inspect local runtime environment")
    research = commands.add_parser("research", help="Run multi-seed WaveForge-X experiments")
    research.add_argument("config", type=Path)
    research.add_argument("--no-cache", action="store_true")
    research.add_argument("--force", action="store_true")
    research.add_argument("--no-plot", action="store_true")
    demo = commands.add_parser("demo", help="Run the reproducible adaptive waveform demonstration")
    demo.add_argument("name", choices=("adaptive",))
    demo.add_argument("--no-cache", action="store_true")
    demo.add_argument("--force", action="store_true")
    demo.add_argument("--no-plot", action="store_true")
    offline = commands.add_parser("train-offline", help="Fit an offline baseline from training trajectories")
    offline.add_argument("directories", nargs="+", type=Path)
    offline.add_argument("--output", required=True, type=Path)
    regions = commands.add_parser("regions", help="Measure and compare waveform operating regions")
    regions.add_argument("config", type=Path)
    regions.add_argument("--output-dir", required=True, type=Path)
    regions.add_argument("--offline-model", type=Path)
    regions.add_argument("--occupancy-run", type=Path)
    regions.add_argument("--results-dir", type=Path)
    sub = commands.add_parser("select", help="Train or plot a measured-region selector")
    selectors = sub.add_subparsers(dest="select_command", required=True)
    train = selectors.add_parser("train")
    train.add_argument("directories", type=Path, nargs="+")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--objective", choices=("ber", "utility"), default="ber")
    train.add_argument("--complexity-weight", type=float, default=.05)
    train.add_argument("--papr-weight", type=float, default=.02)
    plot = selectors.add_parser("map")
    plot.add_argument("model", type=Path)
    plot.add_argument("--output-dir", type=Path, required=True)
    plot.add_argument("--delay-span-s", type=float)
    plot.add_argument("--modulation-bits", type=int)
    args = parser.parse_args(argv)
    try:
        if args.command in ("run", "sweep"):
            from .config import load_config
            from .experiments.runner import run_experiment
            print(run_experiment(load_config(args.config), plot=not args.no_plot))
        elif args.command == "plot":
            if (args.directory / "trajectory.csv").exists():
                from .plotting.dynamic import plot_dynamic
                print(plot_dynamic(args.directory, args.output_dir))
            elif (args.directory / "runs.json").exists():
                from .plotting.research import plot_research
                print(plot_research(args.directory, args.output_dir))
            else:
                from .plotting.publication import plot_results
                print(plot_results(args.directory, args.output_dir))
        elif args.command in ("research", "demo"):
            from .config_dynamic import load_dynamic_config
            from .experiments.research import run_research
            from .reproducibility import project_root
            path = args.config if args.command == "research" else project_root() / "configs/adaptive_demo.yaml"
            print(run_research(load_dynamic_config(path), plot=not args.no_plot,
                               use_cache=not args.no_cache, force=args.force))
        elif args.command == "train-offline":
            from .decision.offline import train_offline
            train_offline(args.directories, args.output)
            print(args.output)
        elif args.command == "regions":
            from .experiments.regions import waveform_regions
            print(waveform_regions(args.config, args.output_dir, offline_model=args.offline_model,
                                    occupancy_run=args.occupancy_run, results_dir=args.results_dir))
        elif args.command == "validate":
            import numpy as np

            from .core.validation import run_validations
            from .waveforms import create_waveform
            results = run_validations()
            x = np.random.default_rng(0).normal(size=64) + 1j * np.random.default_rng(1).normal(size=64)
            for name in ("ofdm", "otfs", "afdm"):
                waveform = create_waveform(name, 64, 8, 8)
                error = float(np.max(np.abs(waveform.demodulate(waveform.modulate(x)) - x)))
                if error > 1e-12:
                    raise ValueError(f"{name} round-trip error {error}")
                results[f"{name}_roundtrip_max_error"] = error
            print(json.dumps(results, indent=2))
        elif args.command == "doctor":
            status = doctor()
            print(json.dumps(status, indent=2))
            return 0 if (status["python_supported"] and all(status["dependencies"].values())
                         and all(status["writable_directories"].values())) else 1
        elif args.select_command == "train":
            from .analysis.waveform_selector import train_selector
            model = train_selector(args.directories, args.objective,
                                   args.complexity_weight, args.papr_weight)
            model.save(args.output)
            print(args.output)
        else:
            from .analysis.waveform_selector import NearestRegionSelector
            from .plotting.publication import plot_selection_map
            print(plot_selection_map(NearestRegionSelector.load(args.model), args.output_dir,
                                     args.delay_span_s, args.modulation_bits))
    except (ValueError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0
