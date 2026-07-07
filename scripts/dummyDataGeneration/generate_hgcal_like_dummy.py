import argparse
import glob
import json
import os.path as osp
import sys

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
sys.path.insert(0, osp.join(REPO_ROOT, "tracksterLinker"))

from tracksterLinker.utils.hgcalDummy import HGCALLikeDummyConfig, summarise_parquet_files, write_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate public HGCAL-like dummy data for trackster-linking experiments.")
    parser.add_argument("--output-dir", default="data/hgcal_like_dummy", help="Directory containing train/val/test parquet files.")
    parser.add_argument(
        "--scenario",
        default="mixed",
        choices=["mixed", "closeby_pions", "multiparticle", "single_particle_pu"],
        help="Synthetic topology to generate.",
    )
    parser.add_argument("--train-files", type=int, default=80)
    parser.add_argument("--val-files", type=int, default=20)
    parser.add_argument("--test-files", type=int, default=20)
    parser.add_argument("--events-per-file", type=int, default=10)
    parser.add_argument("--signal-mean", type=float, default=12.0, help="Mean number of hard-scatter showers for mixed events.")
    parser.add_argument("--pu-mean", type=float, default=35.0, help="Mean number of PU showers for PU-like events.")
    parser.add_argument("--close-pair-fraction", type=float, default=0.35, help="Fraction of hard showers generated near another shower.")
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main():
    args = parse_args()
    config = HGCALLikeDummyConfig(
        scenario=args.scenario,
        train_files=args.train_files,
        val_files=args.val_files,
        test_files=args.test_files,
        events_per_file=args.events_per_file,
        signal_mean=args.signal_mean,
        pu_mean=args.pu_mean,
        close_pair_fraction=args.close_pair_fraction,
        seed=args.seed,
    )
    write_dataset(args.output_dir, config)

    summary = {
        split: summarise_parquet_files(glob.glob(osp.join(args.output_dir, split, "*.parquet")))
        for split in ["train", "val", "test"]
    }
    with open(osp.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
