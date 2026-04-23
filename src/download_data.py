"""
Download NIH ChestX-ray14 dataset from HuggingFace.

Usage:
    python src/download_data.py

The full dataset is ~45GB. This script streams it so you can start
working before the entire download completes. By default it downloads
a small subset first so you can iterate quickly.
"""

from datasets import load_dataset
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download the full dataset. Default downloads a 5k sample for dev.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./data",
        help="Where to cache the dataset.",
    )
    args = parser.parse_args()

    print("Loading NIH ChestX-ray14 from HuggingFace...")
    print(f"Cache dir: {args.cache_dir}")

    if args.full:
        print("Downloading FULL dataset — this will take a while (~45GB)...")
        ds = load_dataset(
            "alkzar90/NIH-Chest-X-ray-dataset",
            "image-classification",
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )
    else:
        print("Downloading a 5000-sample dev split (use --full for everything)...")
        ds = load_dataset(
            "alkzar90/NIH-Chest-X-ray-dataset",
            "image-classification",
            split="train[:5000]",
            cache_dir=args.cache_dir,
            trust_remote_code=True,
        )

    print("\nDataset info:")
    print(ds)
    print("\nSample record:")
    print(ds[0] if not args.full else ds["train"][0])
    print("\nDone! Dataset is ready for exploring.")


if __name__ == "__main__":
    main()
