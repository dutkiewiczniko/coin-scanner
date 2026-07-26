"""Command line interface.

    euro-vision scan data/raw/tray_01.jpg
    euro-vision scan data/raw/ --config my.yaml --verbose
    euro-vision db init
    euro-vision db seed data/rare_coins_seed.csv
    euro-vision db list --denomination 200
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from .config import load_config
from .db import RareCoinStore
from .pipeline import Pipeline
from .results import write_results
from .types import format_denomination

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

log = logging.getLogger("euro_vision")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="euro-vision", description="Euro coin scanning pipeline"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="per-stage logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="run the pipeline over tray image(s)")
    scan.add_argument("path", help="tray image, or a directory of them")
    scan.add_argument("-c", "--config", help="path to a YAML config file")
    scan.add_argument("-o", "--output", help="override output_dir from config")
    scan.add_argument(
        "--no-crops", action="store_true", help="skip writing per-coin crops"
    )
    scan.set_defaults(func=cmd_scan)

    calibrate = subparsers.add_parser(
        "calibrate", help="check corner marker detection on a tray photo"
    )
    calibrate.add_argument("path", help="tray image to check")
    calibrate.add_argument("-c", "--config", help="path to a YAML config file")
    calibrate.add_argument(
        "-o", "--output", help="write the rectified top-down image here"
    )
    calibrate.add_argument(
        "-s",
        "--side",
        choices=["a", "b"],
        help="which tray this photo shows (default: from the filename suffix)",
    )
    calibrate.set_defaults(func=cmd_calibrate)

    ingest = subparsers.add_parser(
        "ingest",
        help="rename phone photos to <batch>_a / <batch>_b using marker winding",
    )
    ingest.add_argument("paths", nargs="+", help="the two photos of one batch")
    ingest.add_argument("-b", "--batch", required=True, help="batch number, e.g. 101")
    ingest.add_argument("-c", "--config", help="path to a YAML config file")
    ingest.add_argument(
        "-d", "--dest", default="data/raw", help="destination directory"
    )
    ingest.add_argument(
        "--move", action="store_true", help="move instead of copying"
    )
    ingest.set_defaults(func=cmd_ingest)

    measure = subparsers.add_parser(
        "measure", help="measure coin diameters and check the scale is true"
    )
    measure.add_argument("path", help="tray image containing coins of known size")
    measure.add_argument("-c", "--config", help="path to a YAML config file")
    measure.add_argument(
        "-s", "--side", choices=["a", "b"], help="which tray (default: from filename)"
    )
    measure.set_defaults(func=cmd_measure)

    db = subparsers.add_parser("db", help="manage the rare coin database")
    db.add_argument("-c", "--config", help="path to a YAML config file")
    db_sub = db.add_subparsers(dest="db_command", required=True)

    db_init = db_sub.add_parser("init", help="create the database schema")
    db_init.set_defaults(func=cmd_db_init)

    db_seed = db_sub.add_parser("seed", help="load rare coins from a CSV file")
    db_seed.add_argument("csv_path", help="CSV with the rare_coin columns")
    db_seed.set_defaults(func=cmd_db_seed)

    db_list = db_sub.add_parser("list", help="list rare coins in the database")
    db_list.add_argument(
        "-d", "--denomination", type=int, help="filter by value in cents, e.g. 200"
    )
    db_list.set_defaults(func=cmd_db_list)

    return parser


# -- commands -------------------------------------------------------------


def cmd_scan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.output:
        config.output_dir = args.output
    if args.no_crops:
        config.save_crops = False

    images = _collect_images(Path(args.path))
    if not images:
        print(f"no images found at {args.path}", file=sys.stderr)
        return 1

    pipeline = Pipeline(config)
    log.info("pipeline: %s", pipeline.describe())

    total_coins = 0
    total_flagged = 0
    for image_path in images:
        result = pipeline.run(image_path)
        written = write_results(result, config)
        total_coins += len(result.coins)
        total_flagged += len(result.flagged)

        print(f"{image_path.name}: {len(result.coins)} coins, "
              f"{len(result.flagged)} flagged -> {written['json']}")
        for coin in result.flagged:
            label = format_denomination(coin.denomination) or "unknown"
            print(f"  [{coin.index:03d}] {label}  {coin.rare_notes}")

    if len(images) > 1:
        print(f"\n{len(images)} images: {total_coins} coins, {total_flagged} flagged")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Report what the corner detection sees, before committing to a full scan."""
    import cv2

    from .pipeline import load_image, save_image, side_from_path
    from .stages import compute_calibration

    config = load_config(args.config)
    image = load_image(args.path)
    side = args.side or side_from_path(args.path)
    calibration = compute_calibration(image, config.calibrate, side)

    print(f"{Path(args.path).name}: {image.shape[1]}x{image.shape[0]} px")
    print(f"tray side: {side}"
          + ("" if args.side else "  (from filename; override with --side)"))
    print("\ncorner markers (clockwise as read):")
    contested = []
    for marker in calibration.markers:
        count = calibration.candidate_counts.get(marker.colour, 1)
        note = "" if count == 1 else f"  [{count} blobs of this colour]"
        if count > 3:
            contested.append((marker.colour, count))
        print(f"  {marker.colour:<8} at ({marker.x:7.1f}, {marker.y:7.1f})"
              f"  blob {marker.area:.0f} px{note}")

    if contested:
        print("\nwarning: " + ", ".join(f"{c} has {n} candidate blobs"
                                        for c, n in contested))
        print("  Something else in shot shares that hue — often the background "
              "or the coins.")
        print("  Corners are picked as the largest quadrilateral, which fails if "
              "an impostor")
        print("  lies OUTSIDE the true corners. Prefer magenta or cyan, which no "
              "coin or")
        print("  common surface matches.")

    top, right, bottom, left = calibration.edge_lengths_px()
    print(f"\nedges in photo: top {top:.0f}  right {right:.0f}  "
          f"bottom {bottom:.0f}  left {left:.0f} px")

    tilt = calibration.tilt_ratio()
    verdict = "square-on" if tilt < 1.02 else "tilted — reposition the camera"
    print(f"tilt ratio: {tilt:.3f} ({verdict})")

    print(f"\nrectified output: {calibration.width_px}x{calibration.height_px} px "
          f"at {calibration.pixels_per_mm:g} px/mm")
    print(f"a 2 euro coin (25.75 mm) will measure "
          f"{25.75 * calibration.pixels_per_mm:.0f} px across")
    if side != "a":
        print("this frame is flipped vertically relative to side a; "
              "pairing undoes that")

    if args.output:
        rectified = cv2.warpPerspective(
            image,
            calibration.homography,
            (calibration.width_px, calibration.height_px),
        )
        save_image(args.output, rectified)
        print(f"\nwrote {args.output}")

    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Name a batch's two photos by working out which tray each one shows.

    The trays are marked with opposite winding, so the photo itself says which
    side it is — no need to remember which came off the camera first. A batch
    whose photos both resolve to the same side means one is missing or a tray
    was mis-marked, and that is worth catching at ingest rather than later.
    """
    import shutil

    from .pipeline import load_image
    from .stages import SIDE_A, SIDE_B, CalibrationError, compute_calibration

    config = load_config(args.config)
    if not config.calibrate.enabled:
        print("error: ingest needs calibrate.enabled: true", file=sys.stderr)
        return 1

    resolved: dict[str, Path] = {}
    for raw in args.paths:
        path = Path(raw)
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 1

        image = load_image(path)
        sides = []
        for side in (SIDE_A, SIDE_B):
            try:
                compute_calibration(image, config.calibrate, side)
                sides.append(side)
            except CalibrationError:
                pass

        if len(sides) != 1:
            reason = ("matches neither tray" if not sides
                      else "matches both trays, so the winding is ambiguous")
            print(f"error: {path.name} {reason}", file=sys.stderr)
            return 1

        side = sides[0]
        if side in resolved:
            print(f"error: {path.name} and {resolved[side].name} are both side "
                  f"'{side}'. One tray of the pair is missing.", file=sys.stderr)
            return 1
        resolved[side] = path

    if len(resolved) != 2:
        print(f"error: need one photo of each tray, got {len(resolved)}",
              file=sys.stderr)
        return 1

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for side, source in sorted(resolved.items()):
        target = dest / f"{args.batch}_{side}{source.suffix.lower()}"
        if target.exists():
            print(f"error: {target} already exists", file=sys.stderr)
            return 1
        if args.move:
            shutil.move(str(source), target)
        else:
            shutil.copy2(source, target)
        print(f"{source.name}  ->  {target}")
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    """Measure every coin and report how far the scale is off.

    The homography assumes the corner markers lie in the coins' plane. Markers
    on a rim do not, so the derived scale carries a systematic error. Comparing
    measured diameters against the eight real Euro sizes recovers the
    correction factor without having to model the geometry.
    """
    from .pipeline import Pipeline
    from .stages.classify import DIAMETERS_MM
    from .types import format_denomination

    config = load_config(args.config)
    if not config.calibrate.enabled:
        print("error: measure needs calibrate.enabled: true", file=sys.stderr)
        return 1

    result = Pipeline(config).run(args.path, side=args.side)
    coins = [c for c in result.coins if c.diameter_mm is not None]
    if not coins:
        print("no coins measured — check the segmentation settings", file=sys.stderr)
        return 1

    print(f"{Path(args.path).name}: {len(coins)} coins at "
          f"{result.meta['pixels_per_mm']:.3g} px/mm "
          f"(correction {config.calibrate.scale_correction:g} applied)\n")
    print(f"  {'#':>3}  {'measured':>9}  {'nearest':>9}  {'ref mm':>7}  {'error':>7}")

    ratios = []
    for coin in coins:
        best, gap = min(
            ((d, abs(coin.diameter_mm - mm)) for d, mm in DIAMETERS_MM.items()),
            key=lambda pair: pair[1],
        )
        flag = "" if gap <= 0.5 else "   <- ambiguous"
        print(f"  {coin.index:>3}  {coin.diameter_mm:>7.2f}mm  "
              f"{format_denomination(best):>9}  {DIAMETERS_MM[best]:>7.2f}  "
              f"{coin.diameter_mm - DIAMETERS_MM[best]:>+6.2f}mm{flag}")
        # Only trust an assignment that is unambiguous; the tightest gap between
        # two denominations is 1 mm, so half of that is the usable window.
        if gap <= 0.5:
            ratios.append(DIAMETERS_MM[best] / coin.diameter_mm)

    if not ratios:
        print("\nNo coin matched a denomination closely enough to calibrate from.")
        print("The scale is likely off by more than 0.5 mm — measure one coin by "
              "hand\nand set calibrate.scale_correction to (true mm / measured mm).")
        return 1

    suggested = sum(ratios) / len(ratios)
    spread = max(ratios) - min(ratios)
    print(f"\n{len(ratios)}/{len(coins)} coins matched unambiguously")
    print(f"suggested calibrate.scale_correction: "
          f"{config.calibrate.scale_correction * suggested:.4f}")
    print(f"spread across coins: {spread * 100:.2f}%"
          + ("  (consistent — a pure scale error)" if spread < 0.01
             else "  (inconsistent — suggests lens distortion, not just scale)"))
    return 0


def cmd_db_init(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    with RareCoinStore(config.rare.database) as store:
        store.init_schema()
        print(f"initialised {config.rare.database} ({store.count()} rare coins)")
    return 0


def cmd_db_seed(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"seed file not found: {csv_path}", file=sys.stderr)
        return 1

    with RareCoinStore(config.rare.database) as store:
        store.init_schema()
        before = store.count()
        for row in _read_seed_csv(csv_path):
            store.add_rare_coin(**row)
        added = store.count() - before
        print(f"seeded {added} new rare coin(s); {store.count()} total")
    return 0


def cmd_db_list(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    with RareCoinStore(config.rare.database) as store:
        rows = store.rare_coins(args.denomination)
        if not rows:
            print("no rare coins in the database — run `euro-vision db seed`")
            return 0
        for row in rows:
            value = format_denomination(row["denomination"])
            year = row["year"] or "----"
            mintage = f"{row['mintage']:,}" if row["mintage"] else "unknown mintage"
            print(f"{value:>7}  {row['country']:<12} {year}  {row['name']}  ({mintage})")
        print(f"\n{len(rows)} coin(s)")
    return 0


# -- helpers --------------------------------------------------------------


def _collect_images(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(
            p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
    return [path] if path.exists() else []


def _read_seed_csv(path: Path) -> list[dict]:
    """Read the seed CSV, skipping `#` comment lines and blanking empty cells."""
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    rows = []
    for row in csv.DictReader(lines):
        cleaned = {k: (v.strip() or None) for k, v in row.items() if k}
        for numeric in ("denomination", "year", "mintage"):
            if cleaned.get(numeric) is not None:
                cleaned[numeric] = int(cleaned[numeric])
        if cleaned.get("est_value_eur") is not None:
            cleaned["est_value_eur"] = float(cleaned["est_value_eur"])
        rows.append(cleaned)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
