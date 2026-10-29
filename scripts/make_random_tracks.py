"""
Makes seeded random waypoint tracks for tuning the MPC. The 9 evaluation tracks
(poisson_generated_tracks.npy) are never used for tuning.

    python -m scripts.make_random_tracks --model-dir models/random_track --out results/random_tracks.npy

The tracks are in map units like create_random_track: vertex 0 is (0, 0), the others are integers in
[-mapWidth/2, mapWidth/2) x [-mapLength/2, mapLength/2). Multiply by mapScale to get metres.
"""
import argparse
import os

import numpy as np
import yaml


def random_tracks(n_tracks, n_vertices, terrain_params, seed):
    """Returns (n_tracks, n_vertices, 2) waypoints in map units."""
    rng = np.random.default_rng(seed)
    x_max = terrain_params['mapWidth'] // 2
    y_max = terrain_params['mapLength'] // 2
    tracks = np.zeros((n_tracks, n_vertices, 2))
    tracks[:, 1:, 0] = rng.integers(-x_max, x_max, (n_tracks, n_vertices - 1))
    tracks[:, 1:, 1] = rng.integers(-y_max, y_max, (n_tracks, n_vertices - 1))
    return tracks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="model dir with config/terrain.yaml")
    parser.add_argument("--n-tracks", type=int, default=9)
    parser.add_argument("--n-vertices", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", required=True, help="output .npy file")
    args = parser.parse_args(argv)

    with open(os.path.join(args.model_dir, "config", "terrain.yaml")) as f:
        terrain_params = yaml.safe_load(f)
    tracks = random_tracks(args.n_tracks, args.n_vertices, terrain_params, args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, tracks)


if __name__ == "__main__":
    main()
