from pathlib import Path
import argparse

from hoi.data_tools.data_loader_leica import LeicaData


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Lift manual Leica panorama instance annotations into 3D point clouds "
            "using the existing pano-tile geometry from the Leica loader."
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
        help="Leica setup identifier to process",
    )
    parser.add_argument(
        "--annotation-json",
        type=Path,
        default=None,
        help="Optional explicit path to the panorama annotation JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for the lifted 3D annotations",
    )
    parser.add_argument(
        "--dedup-voxel",
        type=float,
        default=0.005,
        help="World-space voxel size used to de-duplicate overlapping tile points",
    )
    parser.add_argument(
        "--handle-radius-px",
        type=int,
        default=5,
        help="Panorama pixel radius used when only a handle point is available",
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
    manifest_path = leica_data.lift_pano_instance_annotations_to_3d(
        setup=args.setup,
        annotation_json_path=args.annotation_json,
        output_dir=args.output_dir,
        dedup_voxel=args.dedup_voxel,
        handle_radius_px=args.handle_radius_px,
    )

    print(f"[Leica] 3D annotation manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
