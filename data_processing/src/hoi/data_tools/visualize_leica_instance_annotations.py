from pathlib import Path
import argparse

from hoi.data_tools.data_loader_leica import LeicaData


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize lifted Leica instance annotations as colored point clouds on top "
            "of the downsampled Leica point cloud."
        )
    )
    parser.add_argument(
        "base_path",
        type=Path,
        help="Dataset root, for example /data/ikea_recordings",
    )
    parser.add_argument(
        "rec_loc",
        type=str,
        help="Recording location, for example office_1",
    )
    parser.add_argument(
        "--setup",
        type=str,
        default="001",
        help="Leica setup identifier to visualize",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional explicit path to the lifted annotation manifest JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory containing instances.json and points/*.ply",
    )
    parser.add_argument(
        "--instance-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of instance indices to visualize",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Prepare the visualization and print the legend without opening a window",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    leica_data = LeicaData(
        base_path=args.base_path,
        rec_loc=args.rec_loc,
        initial_setup=args.setup,
    )
    result = leica_data.visualize_instance_annotations_3d(
        setup=args.setup,
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        instance_indices=args.instance_indices,
        show=not args.no_show,
    )

    print(f"[Leica] Visualization manifest: {result['manifest_path']}")
    print(f"[Leica] Prepared {len(result['legend'])} colored instance cloud(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
