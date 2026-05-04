from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a MOT-style sequence with a low-motion repeated-frame section "
            "followed by a high-motion random-frame section."
        )
    )
    parser.add_argument(
        "--source-sequence",
        type=Path,
        default=Path("data/raw/MOT17/train/MOT17-02-FRCNN"),
        help="MOT17 sequence directory containing img1/*.jpg.",
    )
    parser.add_argument(
        "--output-sequence",
        type=Path,
        default=Path("data/processed/custom_sequences/MOT17-custom-lowhigh"),
        help="Output sequence directory to create.",
    )
    parser.add_argument(
        "--low-start-index",
        type=int,
        default=0,
        help="Zero-based index of the first source frame used for the repeated low-motion block.",
    )
    parser.add_argument(
        "--low-block-size",
        type=int,
        default=4,
        help="Number of consecutive source frames in the low-motion repeated block.",
    )
    parser.add_argument(
        "--low-repeats",
        type=int,
        default=25,
        help="Number of times to repeat the low-motion block.",
    )
    parser.add_argument(
        "--high-count",
        type=int,
        default=100,
        help="Number of random source frames to append for the high-motion section.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for high-motion frame selection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output sequence if it already exists.",
    )
    return parser.parse_args()


def read_seqinfo(source_sequence: Path):
    seqinfo_path = source_sequence / "seqinfo.ini"
    values = {
        "frameRate": "30",
        "imWidth": "",
        "imHeight": "",
        "imExt": ".jpg",
    }

    if not seqinfo_path.exists():
        return values

    for line in seqinfo_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in values:
            values[key] = value

    return values


def write_seqinfo(output_sequence: Path, sequence_name: str, seq_length: int, source_sequence: Path):
    source_info = read_seqinfo(source_sequence)
    seqinfo = [
        "[Sequence]",
        f"name={sequence_name}",
        "imDir=img1",
        f"frameRate={source_info['frameRate']}",
        f"seqLength={seq_length}",
        f"imWidth={source_info['imWidth']}",
        f"imHeight={source_info['imHeight']}",
        f"imExt={source_info['imExt']}",
        "",
    ]
    (output_sequence / "seqinfo.ini").write_text("\n".join(seqinfo))


def build_frame_plan(image_paths, low_start_index, low_block_size, low_repeats, high_count, seed):
    low_end_index = low_start_index + low_block_size
    if low_start_index < 0 or low_end_index > len(image_paths):
        raise ValueError(
            f"Low-motion block [{low_start_index}:{low_end_index}] is outside "
            f"the {len(image_paths)} available source frames."
        )

    low_block = image_paths[low_start_index:low_end_index]
    low_frames = low_block * low_repeats

    low_set = set(low_block)
    high_candidates = [path for path in image_paths if path not in low_set]
    if high_count > len(high_candidates):
        raise ValueError(
            f"Requested {high_count} high-motion frames, but only "
            f"{len(high_candidates)} candidates are available."
        )

    rng = random.Random(seed)
    high_frames = rng.sample(high_candidates, high_count)

    plan = []
    for source_path in low_frames:
        plan.append(("low_repeated", source_path))
    for source_path in high_frames:
        plan.append(("high_random", source_path))

    return plan


def create_sequence(args):
    source_img_dir = args.source_sequence / "img1"
    if not source_img_dir.exists():
        raise FileNotFoundError(f"Missing source img1 directory: {source_img_dir}")

    image_paths = sorted(source_img_dir.glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"No jpg frames found in {source_img_dir}")

    if args.output_sequence.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output sequence already exists: {args.output_sequence}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(args.output_sequence)

    output_img_dir = args.output_sequence / "img1"
    output_img_dir.mkdir(parents=True, exist_ok=True)

    frame_plan = build_frame_plan(
        image_paths=image_paths,
        low_start_index=args.low_start_index,
        low_block_size=args.low_block_size,
        low_repeats=args.low_repeats,
        high_count=args.high_count,
        seed=args.seed,
    )

    manifest_rows = []
    for output_idx, (phase, source_path) in enumerate(frame_plan, start=1):
        output_name = f"{output_idx:06d}{source_path.suffix.lower()}"
        output_path = output_img_dir / output_name
        shutil.copy2(source_path, output_path)

        manifest_rows.append(
            {
                "generated_frame": output_idx,
                "phase": phase,
                "source_sequence": args.source_sequence.name,
                "source_frame": int(source_path.stem),
                "source_path": str(source_path),
                "output_path": str(output_path),
            }
        )

    manifest_path = args.output_sequence / "frame_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    write_seqinfo(
        output_sequence=args.output_sequence,
        sequence_name=args.output_sequence.name,
        seq_length=len(frame_plan),
        source_sequence=args.source_sequence,
    )

    print(f"Created custom sequence: {args.output_sequence}")
    print(f"Frames written: {len(frame_plan)}")
    print(f"Low-motion repeated frames: {args.low_block_size * args.low_repeats}")
    print(f"High-motion random frames: {args.high_count}")
    print(f"Manifest: {manifest_path}")


def main():
    args = parse_args()
    create_sequence(args)


if __name__ == "__main__":
    main()
