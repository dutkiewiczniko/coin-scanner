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

    export = subparsers.add_parser(
        "export-dataset",
        help="write a YOLO training set from tray photos, pre-labelled",
    )
    export.add_argument("paths", nargs="+", help="tray images, or directories")
    export.add_argument("-c", "--config", help="path to a YAML config file")
    export.add_argument("-o", "--out", default="data/dataset", help="output directory")
    export.add_argument(
        "--by-denomination", action="store_true",
        help="label eight denomination classes instead of a single 'coin' class",
    )
    export.add_argument("--val-split", type=float, default=0.2)
    export.add_argument(
        "--crops", action="store_true",
        help="export per-coin crops in folder-per-class layout for a "
             "denomination classifier, instead of whole-tray detection labels",
    )
    export.set_defaults(func=cmd_export_dataset)

    pair = subparsers.add_parser(
        "pair", help="match both faces of a batch into per-coin images"
    )
    pair.add_argument("batch", help="batch number, e.g. 101")
    pair.add_argument(
        "-s", "--source", default="data/raw",
        help="directory holding <batch>_a and <batch>_b (default: data/raw)",
    )
    pair.add_argument("-c", "--config", help="path to a YAML config file")
    pair.add_argument("-o", "--output", help="override output_dir from config")
    pair.set_defaults(func=cmd_pair)

    measure = subparsers.add_parser(
        "measure", help="measure coin diameters and check the scale is true"
    )
    measure.add_argument("path", help="tray image containing coins of known size")
    measure.add_argument("-c", "--config", help="path to a YAML config file")
    measure.add_argument(
        "-s", "--side", choices=["a", "b"], help="which tray (default: from filename)"
    )
    measure.set_defaults(func=cmd_measure)

    fit_metal = subparsers.add_parser(
        "fit-metal",
        help="fit the alloy model used by the 'metal' classify backend",
    )
    fit_metal.add_argument(
        "crops",
        help="folder of labelled crops, one subfolder per denomination "
        "(1c, 2c, 5c, 10c, 20c, 50c, 1_EUR, 2_EUR) as written by "
        "`export-dataset --crops`",
    )
    fit_metal.add_argument("-c", "--config", help="path to a YAML config file")
    fit_metal.add_argument(
        "-o", "--output", help="where to write the model (default: from config)"
    )
    fit_metal.set_defaults(func=cmd_fit_metal)

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
    db_list.add_argument(
        "-m", "--max-mintage", type=int,
        help="only show issues struck this many times or fewer",
    )
    db_list.add_argument("-n", "--limit", type=int, default=40)
    db_list.set_defaults(func=cmd_db_list)

    db_import = db_sub.add_parser(
        "import-commemoratives",
        help="load every 2 EUR commemorative issue and its mintage from Wikipedia",
    )
    db_import.add_argument(
        "--offline", metavar="JSON",
        help="parse a previously saved wikitext response instead of fetching",
    )
    db_import.set_defaults(func=cmd_db_import_commemoratives)

    numista = subparsers.add_parser(
        "numista", help="query the Numista catalogue for varieties and mintages"
    )
    numista.add_argument("-c", "--config", help="path to a YAML config file")
    n_sub = numista.add_subparsers(dest="numista_command", required=True)

    n_check = n_sub.add_parser("check", help="verify the API key works (1 request)")
    n_check.set_defaults(func=cmd_numista_check)

    n_search = n_sub.add_parser("search", help="search the catalogue by text")
    n_search.add_argument("query", help='e.g. "2 euro Hamburg mule"')
    n_search.add_argument("-n", "--limit", type=int, default=20)
    n_search.set_defaults(func=cmd_numista_search)

    n_type = n_sub.add_parser("type", help="show one type, its issues and mintages")
    n_type.add_argument("type_id", help="Numista N#, e.g. 327881")
    n_type.set_defaults(func=cmd_numista_type)

    n_scarce = n_sub.add_parser(
        "scarce",
        help="sweep an issuer for types struck far less than their same-year "
             "siblings — the signature of a mule or caught die error",
    )
    n_scarce.add_argument(
        "issuer",
        help="Numista issuer code, e.g. allemagne, or a country name like germany",
    )
    n_scarce.add_argument(
        "-q", "--query", default="euro",
        help="restrict the sweep to types matching this text (default: euro)",
    )
    n_scarce.add_argument(
        "--max-types", type=int, default=60,
        help="stop after this many types. Each one costs a request for its "
             "issues, and the free tier is 2,000 a month.",
    )
    n_scarce.add_argument(
        "--ratio", type=float,
        help="override numista.outlier_ratio",
    )
    n_scarce.add_argument(
        "--save", action="store_true",
        help="write the candidates found to the rare coin database",
    )
    n_scarce.add_argument(
        "--yes", action="store_true", help="skip the request-budget prompt"
    )
    n_scarce.set_defaults(func=cmd_numista_scarce)

    n_fetch = n_sub.add_parser(
        "fetch",
        help="resolve a list of target coins against Numista and keep the "
             "scarce ones — the cherry-picking alternative to a full sweep",
    )
    n_fetch.add_argument("targets", help="CSV of targets, e.g. data/numista_targets_200.csv")
    n_fetch.add_argument(
        "-m", "--max-mintage", type=int,
        help="keep issues struck this many times or fewer "
             "(default: rare.scarce_mintage from config)",
    )
    n_fetch.add_argument(
        "--per-query", type=int, default=4,
        help="how many search hits to examine per target row",
    )
    n_fetch.add_argument(
        "--varieties-only", action="store_true",
        help="keep only issues whose own comment describes an error or variety, "
             "ignoring mintage. The right mode below 1 EUR, where value comes "
             "from the fault rather than from scarcity.",
    )
    n_fetch.add_argument(
        "--include-sets", action="store_true",
        help="keep proof and collector-set issues too. Off by default: they "
             "never reach a bank bag, and they outnumber the real finds.",
    )
    n_fetch.add_argument(
        "--save", action="store_true", help="write the keepers to the database"
    )
    n_fetch.add_argument(
        "--yes", action="store_true", help="skip the request-budget prompt"
    )
    n_fetch.set_defaults(func=cmd_numista_fetch)

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


def cmd_fit_metal(args: argparse.Namespace) -> int:
    """Refit the alloy centroids from labelled crops.

    Worth rerunning whenever the lighting or backdrop changes. The features are
    channel ratios rather than absolute colours, which is what lets copper and
    gold separate under warm light where raw hue could not, but a different
    light source still moves them.
    """
    import json

    from .metal import METAL_GROUP, GroupModel, metal_features
    from .pipeline import load_image
    from .types import DENOMINATIONS, format_denomination

    config = load_config(args.config)
    root = Path(args.crops)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    by_label = {
        (format_denomination(d) or "").replace(" ", "_"): d for d in DENOMINATIONS
    }
    features, groups, counts = [], [], {}
    # Accepts either a flat folder of class directories or the train/val layout
    # that `export-dataset --crops` writes.
    for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
        cents = by_label.get(folder.name)
        if cents is None:
            continue
        group = METAL_GROUP[cents]
        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            features.append(metal_features(load_image(image_path)))
            groups.append(group)
            counts[group] = counts.get(group, 0) + 1

    if len(set(groups)) < 2:
        print(
            f"error: found {len(groups)} labelled crops covering "
            f"{sorted(set(groups))}; need at least two alloys",
            file=sys.stderr,
        )
        return 1

    model = GroupModel.fit(features, groups)
    out = Path(args.output or config.classify.metal_model)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = model.to_dict()
    payload["note"] = f"Fitted from {root} on {len(groups)} crops: {counts}"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(f"fitted on {len(groups)} crops {counts}")
    print(f"  -> {out}")
    if min(counts.values()) < 10:
        thin = [g for g, n in counts.items() if n < 10]
        print(
            f"  note: only {min(counts.values())} example(s) for {thin}; "
            "that centroid will move a lot with a few more coins"
        )
    return 0


def cmd_export_dataset(args: argparse.Namespace) -> int:
    from .dataset import export_dataset

    config = load_config(args.config)
    paths: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            paths.extend(
                p for p in sorted(path.iterdir())
                if p.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.exists():
            paths.append(path)
        else:
            print(f"error: {path} not found", file=sys.stderr)
            return 1

    if not paths:
        print("error: no images found", file=sys.stderr)
        return 1

    if not config.calibrate.enabled:
        print("note: calibration is off, so crops come from the photo as shot "
              "and no\n      diameters are measured. Fine for a classifier that "
              "reads the coin;\n      not fine for size-based classification.\n")

    out = Path(args.out)
    if args.crops:
        from .dataset import export_crops

        stats = export_crops(paths, out, config, val_split=args.val_split)
        noun = "crops"
    else:
        stats = export_dataset(
            paths, out, config,
            by_denomination=args.by_denomination,
            val_split=args.val_split,
        )
        noun = "labelled coins"

    print(f"{stats.images} images, {stats.instances} {noun} -> {out}")
    for name, count in sorted(stats.per_class.items(), key=lambda kv: -kv[1]):
        print(f"  {name:>10}: {count}")
    for problem in stats.skipped:
        print(f"  skipped {problem}", file=sys.stderr)

    if args.crops:
        unsorted = stats.per_class.get("_unsorted", 0)
        if unsorted:
            print(f"\n  {unsorted} crops could not be labelled confidently and "
                  f"went to _unsorted/.")
            print("  Sorting those into class folders is the labelling work.")
        print("\n  This is folder-per-class layout, so public coin datasets in "
              "the same\n  shape can be merged by copying their class folders in.")
    elif args.by_denomination:
        thin = [n for n, c in stats.per_class.items() if c < 100]
        if thin:
            print(f"\n  Thin classes (<100 instances): {', '.join(sorted(thin))}")
            print("  Detection tolerates small datasets; per-class appearance "
                  "does not.")

    print("\n  Labels come from this pipeline's own measurements. Correct them "
          "before\n  training — a model fitted to uncorrected output can only "
          "repeat its mistakes.")
    return 0


def cmd_pair(args: argparse.Namespace) -> int:
    """Scan both trays of a batch and match each coin's two faces."""
    from .pairing import pair_batch, write_batch
    from .pipeline import Pipeline

    config = load_config(args.config)
    if args.output:
        config.output_dir = args.output
    if not config.calibrate.enabled:
        print("error: pair needs calibrate.enabled: true — both faces have to "
              "land in the same metric frame", file=sys.stderr)
        return 1

    source = Path(args.source)
    paths = {}
    for side in ("a", "b"):
        found = [
            p for p in sorted(source.iterdir())
            if p.stem.lower() == f"{args.batch}_{side}"
            and p.suffix.lower() in IMAGE_SUFFIXES
        ] if source.is_dir() else []
        if not found:
            print(f"error: no {args.batch}_{side} image in {source} — run "
                  f"`euro-vision ingest` first", file=sys.stderr)
            return 1
        paths[side] = found[0]

    pipeline = Pipeline(config)
    result_a = pipeline.run(paths["a"], side="a")
    result_b = pipeline.run(paths["b"], side="b")

    batch = pair_batch(args.batch, result_a, result_b, config)
    written = write_batch(batch, config)

    total = len(batch.coins)
    paired = len(batch.paired)
    print(f"batch {args.batch}: {len(result_a.coins)} coins on side a, "
          f"{len(result_b.coins)} on side b")
    print(f"  {paired} matched, {len(batch.unpaired)} unmatched "
          f"-> {total} coin records")
    if paired:
        offsets = sorted(c.offset_mm for c in batch.paired)
        print(f"  match offset: median {offsets[len(offsets)//2]:.2f} mm, "
              f"worst {offsets[-1]:.2f} mm")

        # Both faces of one coin should classify the same way. This is not proof
        # a match is right — the classifier has its own error — but a low rate
        # means either matching or measurement is going wrong, and it is the
        # only check available without labelling by hand.
        agree = sum(
            1 for c in batch.paired
            if c.face_a.denomination == c.face_b.denomination
        )
        print(f"  both faces agree on denomination: {agree}/{paired} "
              f"({agree / paired:.0%})")

    flagged = [c for c in batch.coins if c.coin.is_flagged]
    if flagged:
        print(f"\n  {len(flagged)} flagged for review:")
        for c in flagged[:10]:
            print(f"    {c.coin_id}  "
                  f"{format_denomination(c.coin.denomination) or 'unknown':>7}  "
                  f"at {c.x_mm:.0f},{c.y_mm:.0f} mm")
        if len(flagged) > 10:
            print(f"    ... and {len(flagged) - 10} more")

    print(f"\n  {written['json']}")
    print(f"  {written['coins']}/")
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

    print(f"\n{len(ratios)}/{len(coins)} coins matched a denomination "
          f"unambiguously (within 0.5 mm)")

    # Fitting the whole population, rather than averaging per-coin ratios, is
    # the part that can detect a large error. Nearest-reference matching is
    # self-confirming: if every coin reads a denomination too small, each one
    # still sits close to *a* real diameter and the scale looks correct.
    import numpy as np

    measured = np.array([c.diameter_mm for c in coins])
    reference = np.array(sorted(DIAMETERS_MM.values()))
    factors = np.arange(0.90, 1.101, 0.002)
    errors = [
        np.abs((measured * f)[:, None] - reference[None, :]).min(axis=1).mean()
        for f in factors
    ]
    best = float(factors[int(np.argmin(errors))])
    at_one = errors[int(np.argmin(np.abs(factors - 1.0)))]

    print(f"\nbest-fit scale over all {len(coins)} coins: {best:.4f} "
          f"(mean error {min(errors):.3f} mm, vs {at_one:.3f} mm as-is)")
    print(f"  -> calibrate.scale_correction "
          f"{config.calibrate.scale_correction * best:.4f}")

    if abs(best - 1.0) > 0.01:
        print(f"\n  Scale looks {abs(1 - best) * 100:.1f}% out. Check "
              f"calibrate.tray_width_mm / tray_height_mm against the actual")
        print(f"  marker centre-to-centre distance — that is the likeliest "
              f"cause, and this")
        print(f"  fit is too shallow to replace a real measurement.")

    if ratios:
        spread = max(ratios) - min(ratios)
        print(f"\nspread across confidently matched coins: {spread * 100:.2f}%"
              + ("  (consistent — a pure scale error)" if spread < 0.01
                 else "  (inconsistent — lens distortion or measurement noise)"))
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
            print("no rare coins in the database — run "
                  "`euro-vision db import-commemoratives`")
            return 0
        limit = getattr(args, "max_mintage", None)
        if limit is not None:
            rows = [r for r in rows if r["mintage"] is not None
                    and r["mintage"] <= limit]
        # Scarcest first: the point of the list is to know what to look for.
        rows = sorted(rows, key=lambda r: (r["mintage"] is None, r["mintage"] or 0))
        shown = rows[: args.limit]
        for row in shown:
            value = format_denomination(row["denomination"])
            year = row["year"] or "----"
            mintage = f"{row['mintage']:,}" if row["mintage"] else "unknown mintage"
            print(f"{mintage:>12}  {value:>6}  {row['country']:<22} "
                  f"{year}  {row['name'][:52]}")
        if len(rows) > len(shown):
            print(f"... and {len(rows) - len(shown)} more")
        print(f"\n{len(rows)} coin(s)")
    return 0


def cmd_db_import_commemoratives(args: argparse.Namespace) -> int:
    """Load every 2 EUR commemorative issue, common and scarce alike.

    Everything is stored rather than only the scarce ones so that rarity stays a
    query against mintage instead of a list someone has to keep up to date. An
    issue that is common today is still the row you need in order to say so.
    """
    import json as _json

    from .catalogue import (
        COMMEMORATIVE_DENOMINATION,
        SOURCE_URL,
        parse_issues,
        load_commemoratives,
    )

    config = load_config(args.config)
    try:
        if args.offline:
            wikitext = _json.loads(
                Path(args.offline).read_text(encoding="utf-8")
            )["parse"]["wikitext"]
            issues = parse_issues(wikitext)
        else:
            issues = load_commemoratives()
    except Exception as exc:  # noqa: BLE001 - network and parsing both surface here
        print(f"error: could not load the catalogue: {exc}", file=sys.stderr)
        return 1

    if not issues:
        print("error: parsed no issues — the page layout may have changed",
              file=sys.stderr)
        return 1

    threshold = config.rare.scarce_mintage
    added = 0
    with RareCoinStore(config.rare.database) as store:
        store.init_schema()
        for issue in issues:
            store.add_rare_coin(
                country=issue.country,
                denomination=COMMEMORATIVE_DENOMINATION,
                year=issue.year,
                variant="commemorative",
                name=issue.feature or issue.name,
                mintage=issue.mintage,
                notes=issue.description[:500],
                source_url=SOURCE_URL,
            )
            added += 1
        total = store.count()

    missing = sum(1 for i in issues if i.mintage is None)
    scarce = sum(1 for i in issues
                 if i.mintage is not None and i.mintage <= threshold)
    print(f"loaded {added} commemorative issues "
          f"({min(i.year for i in issues)}-{max(i.year for i in issues)}), "
          f"{total} rows in the database")
    if missing:
        print(f"  {missing} without a mintage figure — left NULL rather than guessed")
    print(f"  {scarce} at or below the {threshold:,} scarcity threshold")
    print("  mintages are Official Journal planned volumes; actual production "
          "can deviate")
    return 0


# -- numista --------------------------------------------------------------


def _numista(args: argparse.Namespace):
    """Build a client from config, with the key read from the environment."""
    import os

    from .numista import Numista

    config = load_config(args.config).numista
    return Numista(
        api_key=os.environ.get(config.api_key_env),
        lang=config.lang,
        cache_dir=config.cache_dir,
        cache_days=config.cache_days,
        min_interval_s=config.min_interval_s,
    ), config


def cmd_numista_check(args: argparse.Namespace) -> int:
    from .numista import API_KEY_ENV, KEY_URL, NumistaError

    client, config = _numista(args)
    if not client.api_key:
        print(f"no API key: ${config.api_key_env} is not set.\n")
        print("  1. Log in to Numista, then open")
        print(f"       {KEY_URL}")
        print("     Free, 2,000 requests/month. That page is a button on "
              "/api/index.php with")
        print("     no href, so it cannot be found by browsing — go straight "
              "to the URL.")
        print(f"  2. $env:{config.api_key_env} = '<key>'   "
              f"(PowerShell; export {API_KEY_ENV}=<key> elsewhere)")
        print("\nThe key is read from the environment and never written to the "
              "config file,\nwhich is committed.")
        return 1

    try:
        payload = client.check()
    except NumistaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"connected to Numista as key ...{client.api_key[-4:]} "
          f"({payload.get('count', 0):,} coin types match 'euro')")
    print(f"  cache: {config.cache_dir} ({config.cache_days} day TTL)")
    print(f"  spent: {client.requests_made} request(s) this run")
    return 0


def cmd_numista_search(args: argparse.Namespace) -> int:
    from .numista import NumistaError

    client, _ = _numista(args)
    try:
        total, types = client.search_types(args.query, count=min(args.limit, 50))
    except NumistaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{total:,} match(es) for {args.query!r}, showing {len(types)}\n")
    for t in types:
        years = (f"{t.min_year}-{t.max_year}" if t.min_year != t.max_year
                 else str(t.min_year or ""))
        refs = f"  [{'; '.join(t.references)}]" if t.references else ""
        print(f"  {t.n_number:<10} {years:<10} {t.title[:60]}{refs}")
        print(f"  {'':<10} {t.issuer}  {t.url}")
    return 0


def cmd_numista_type(args: argparse.Namespace) -> int:
    """Show a type and its per-year mintages.

    Two requests: the type record, then its issues. The comment is printed in
    full because on a variety entry that is where the mint's mistake is
    described — but see `numista.py` on why nothing parses it.
    """
    from .numista import NumistaError, plain_text, variety_hint

    client, _ = _numista(args)
    try:
        record = client.get_type(args.type_id)
        issues = client.get_issues(args.type_id)
    except NumistaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    issuer = (record.get("issuer") or {}).get("name", "")
    print(f"N# {record.get('id')}  {record.get('title', '')}")
    print(f"  issuer:      {issuer}")
    print(f"  years:       {record.get('min_year')} - {record.get('max_year')}")
    composition = (record.get("composition") or {}).get("text")
    if composition:
        print(f"  composition: {composition}")
    for key in ("weight", "size", "thickness"):
        if record.get(key) is not None:
            print(f"  {key + ':':<13}{record[key]}")
    refs = "; ".join(
        f"{(r.get('catalogue') or {}).get('code', '')}# {r.get('number', '')}"
        for r in record.get("references", []) or []
    )
    if refs:
        print(f"  references:  {refs}")
    print(f"  url:         {record.get('url')}")

    # The field is `comments`, plural, and it is HTML — bold tags, &quot;
    # entities, and sometimes an <img> of the variety.
    comment = plain_text(record.get("comments") or "")
    if comment:
        hint = variety_hint(comment)
        print(f"\ncomments{f'  (variety words: {hint})' if hint else ''}:")
        print("  " + comment.replace("\n", "\n  "))

    if issues:
        print(f"\nissues ({len(issues)}):")
        for issue in issues:
            mintage = f"{issue.mintage:,}" if issue.mintage is not None else "unknown"
            mark = f" ({issue.mint_letter})" if issue.mint_letter else ""
            note = f"  {issue.comment}" if issue.comment else ""
            print(f"  NN# {issue.id:<8} {issue.year or '----'}{mark:<5} "
                  f"{mintage:>13}{note}")

    print(f"\n  {client.requests_made} request(s) spent")
    return 0


def cmd_numista_scarce(args: argparse.Namespace) -> int:
    """Find types struck far less than their same-year siblings.

    This is the mechanical version of "look for the coin whose description
    mentions an error". A mule or a caught die fault becomes its own catalogue
    type with its own, small, mintage, sitting beside the corrected type struck
    in the millions. That ratio is language-independent and does not depend on
    anyone having written the word "mule" anywhere.

    Costs one request per page of types plus one per type for its issues, which
    is why it asks before it starts.
    """
    from .numista import (
        EURO_ISSUERS,
        NumistaError,
        mintage_outliers,
        plain_text,
        variety_hint,
    )

    client, config = _numista(args)
    issuer = EURO_ISSUERS.get(args.issuer.lower().replace(" ", "_"), args.issuer)

    budget = args.max_types + (args.max_types // 50) + 1
    if not args.yes:
        print(f"sweeping issuer {issuer!r} for up to {args.max_types} types.")
        print(f"  up to {budget} API requests, against a free tier of 2,000 a "
              f"month.")
        print(f"  cached responses in {config.cache_dir} are reused and cost "
              f"nothing.")
        if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted")
            return 0

    try:
        types = list(client.all_types(
            q=args.query or None,
            issuer=issuer,
            max_requests=max(1, args.max_types // 50 + 1),
        ))[: args.max_types]
        if not types:
            print(f"no types found for issuer {issuer!r} matching "
                  f"{args.query!r}. Check the code against "
                  f"`euro-vision numista search`.", file=sys.stderr)
            return 1
        typed_issues = [(t, client.get_issues(t.id)) for t in types]
    except NumistaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted — partial results are cached and cost nothing to "
              "resume", file=sys.stderr)
        return 1

    candidates = mintage_outliers(
        typed_issues, ratio=args.ratio or config.outlier_ratio
    )
    with_mintage = sum(
        1 for _, issues in typed_issues
        for i in issues if i.mintage is not None
    )
    print(f"\n{len(types)} types, {with_mintage} issues carrying a mintage, "
          f"{client.requests_made} request(s) spent")

    if not candidates:
        print("\nno mintage outliers. Either this issuer has no catalogued "
              "varieties, or\nthe sweep did not reach them — raise --max-types "
              "or loosen --ratio.")
        return 0

    print(f"\n{len(candidates)} candidate(s), most anomalous first:\n")
    for c in candidates:
        print(f"  {c.type.n_number:<10} {c.year}  {c.type.title[:56]}")
        print(f"  {'':<10} {c.mintage:,} struck, {c.ratio:.1%} of the "
              f"{c.sibling_mintage:,} sibling")
        if c.type.references:
            print(f"  {'':<10} {'; '.join(c.type.references)}")
        print(f"  {'':<10} {c.type.url}")

    print("\n  A low sibling mintage also means a proof-only issue or a short "
          "commemorative\n  run, so this is a shortlist to look at, not a "
          "verdict. Open the N# and read\n  the comment to see which it is.")

    if args.save:
        # The variety text can sit on either the type or the issue, and on
        # N# 327881 — the Hamburg mule — the type comment is empty while the
        # issue carries "Error: old map style on reverse (Mule)". Reading only
        # the type would have filed it as an ordinary standard issue.
        issue_notes = {
            t.id: " ".join(i.comment for i in issues if i.comment)
            for t, issues in typed_issues
        }
        full_config = load_config(args.config)
        with RareCoinStore(full_config.rare.database) as store:
            store.init_schema()
            before = store.count()
            for c in candidates:
                record = client.get_type(c.type.id)
                comment = " ".join(
                    part for part in (plain_text(record.get("comments") or ""),
                                      issue_notes.get(c.type.id, ""))
                    if part
                ).strip()
                hint = variety_hint(comment)
                store.add_rare_coin(
                    country=c.type.issuer,
                    denomination=_denomination_from_title(c.type.title),
                    year=c.year,
                    variant="error" if hint else "standard",
                    name=f"{c.type.title} ({c.type.n_number})"[:120],
                    mintage=c.mintage,
                    notes=(f"{c.reason}. {comment}").strip()[:500],
                    source_url=c.type.url,
                )
            print(f"\n  saved {store.count() - before} new row(s) to "
                  f"{full_config.rare.database} "
                  f"({client.requests_made} requests total)")
    return 0


def cmd_numista_fetch(args: argparse.Namespace) -> int:
    """Resolve a curated target list and keep only the scarce results.

    Cherry-picking rather than mirroring. Each CSV row is a *search*, not a
    fact — the mintage always comes back from Numista, so nothing enters the
    database on the strength of someone's memory, and a target that turns out
    to be common is dropped instead of being recorded as rare.

    Costs one request per row, plus one per type examined, plus one per type
    kept. A 21-row list at --per-query 4 is roughly 150 requests against the
    free tier's 2,000 a month.
    """
    from .numista import (
        CIRCULATING_QUALITIES,
        EURO_ISSUERS,
        NumistaError,
        denomination_cents,
        issue_quality,
        plain_text,
        variant_of,
        variety_hint,
    )

    full_config = load_config(args.config)
    client, _ = _numista(args)
    threshold = args.max_mintage or full_config.rare.scarce_mintage

    targets = _read_targets(Path(args.targets))
    if not targets:
        print(f"error: no target rows in {args.targets}", file=sys.stderr)
        return 1

    estimate = len(targets) * (1 + args.per_query)
    if not args.yes:
        print(f"{len(targets)} target(s) from {args.targets}")
        print(f"  keeping issues struck {threshold:,} times or fewer")
        print(f"  up to ~{estimate} API requests (free tier: 2,000/month); "
              f"cached rows cost nothing")
        if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted")
            return 0

    keepers: dict[tuple[int, int], dict] = {}
    seen_types: set[int] = set()
    misses: list[str] = []
    skipped_quality: dict[str, int] = {}

    try:
        for row in targets:
            issuer = EURO_ISSUERS.get(
                (row["issuer"] or "").lower().replace(" ", "_"), row["issuer"] or None
            )
            # One bad row must not discard the whole run. An invalid issuer code
            # is a hard HTTP 400, and letting that propagate threw away every
            # result already paid for — the requests are spent either way, so
            # the only thing aborting achieves is wasting them.
            try:
                _, hits = client.search_types(
                    row["query"], issuer=issuer, count=args.per_query
                )
            except NumistaError as exc:
                misses.append(f"{row['query']!r} ({issuer or 'any issuer'}): {exc}")
                continue
            if not hits:
                misses.append(f"{row['query']!r} ({issuer or 'any issuer'}): no match")
                continue

            for summary in hits[: args.per_query]:
                if summary.id in seen_types:
                    continue
                seen_types.add(summary.id)
                try:
                    issues = client.get_issues(summary.id)
                except NumistaError as exc:
                    misses.append(f"{summary.n_number} issues: {exc}")
                    continue
                for issue in issues:
                    is_variety = bool(variety_hint(issue.comment))
                    # A variety is kept whatever its mintage, including when
                    # that is unknown. Italy's 2002 1c "Error - missing R
                    # mintmark" has no figure at all — nobody counted them —
                    # and skipping unmintaged issues threw away exactly the
                    # coins this is looking for. The Hamburg mule makes the
                    # same point from the other end: 600,000 is common by
                    # mintage and still worth finding.
                    if not is_variety:
                        # Below 1 EUR scarcity is worth nothing on its own — a
                        # 1c struck 5,000 times is still worth a cent — so
                        # --varieties-only drops the mintage test entirely
                        # rather than merely tightening it.
                        if args.varieties_only:
                            continue
                        if issue.mintage is None or issue.mintage > threshold:
                            continue
                    quality = issue_quality(issue.comment)
                    if not args.include_sets and quality not in CIRCULATING_QUALITIES:
                        skipped_quality[quality] = skipped_quality.get(quality, 0) + 1
                        continue
                    record = client.get_type(summary.id)
                    cents = denomination_cents(record)
                    if cents is None:
                        # Not a euro coin — the search wandered. Better dropped
                        # than stored with a guessed denomination.
                        continue
                    if row["denomination"] and cents != row["denomination"]:
                        continue
                    comments = plain_text(record.get("comments") or "")
                    # Two different strengths of evidence, and collapsing them
                    # into one flag is wrong. Finland's N# 95 carries "Die error
                    # on the 2000 coin" in its *type* comments, which said
                    # nothing about 1999 or 2003 — yet stamped all eight years
                    # as varieties. The issue comment is per-coin and is the
                    # only one that justifies the claim.
                    keepers[(summary.id, issue.id)] = {
                        "summary": summary,
                        "issue": issue,
                        "cents": cents,
                        "quality": quality,
                        "variant": variant_of(record.get("type"), is_variety),
                        "comments": comments,
                        "hint": variety_hint(issue.comment),
                        "type_hint": variety_hint(comments),
                        "note": row["note"],
                    }
    except NumistaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted — what was fetched is cached and costs nothing to "
              "resume", file=sys.stderr)
        return 1

    varieties = sum(1 for e in keepers.values() if e["hint"])
    if args.varieties_only:
        print(f"\n{len(seen_types)} types examined, {len(keepers)} variety "
              f"issue(s) kept (mintage ignored), "
              f"{client.requests_made} request(s) spent")
    else:
        # Only those over the threshold were kept *because* they are varieties;
        # counting all flagged keepers overstates what the rule contributed.
        by_variety = sum(
            1 for e in keepers.values()
            if e["hint"] and (e["issue"].mintage or 0) > threshold
        )
        print(f"\n{len(seen_types)} types examined, {len(keepers)} issue(s) kept "
              f"at or under {threshold:,} ({varieties} of them varieties; "
              f"{by_variety} kept on the variety rule alone), "
              f"{client.requests_made} request(s) spent")
    if skipped_quality:
        detail = ", ".join(f"{n} {q}" for q, n in sorted(skipped_quality.items()))
        print(f"  dropped {sum(skipped_quality.values())} collector-only "
              f"issue(s): {detail}  (--include-sets keeps them)")
    if misses:
        print(f"\n  {len(misses)} target(s) matched nothing:")
        for miss in misses:
            print(f"    {miss}")

    if not keepers:
        print("\nnothing kept. Raise --max-mintage, or the issuer codes may be "
              "wrong.")
        return 0

    print()
    print(f"  {'mintage':>10}  {'value':>6}  {'year':<10} {'quality':<12} "
          f"{'country':<24} title")
    # Unknown mintage sorts last, not as zero — it is absent, not tiny.
    for entry in sorted(
        keepers.values(),
        key=lambda e: (e["issue"].mintage is None, e["issue"].mintage or 0),
    ):
        issue, summary = entry["issue"], entry["summary"]
        mark = f" ({issue.mint_letter})" if issue.mint_letter else ""
        flag = ("  <- VARIETY" if entry["hint"]
                else "  (type notes errors in some years)" if entry["type_hint"]
                else "")
        mintage = f"{issue.mintage:,}" if issue.mintage is not None else "unknown"
        print(f"  {mintage:>10}  "
              f"{format_denomination(entry['cents']):>6}  "
              f"{str(issue.year or '----') + mark:<10} "
              f"{entry['quality']:<12} {summary.issuer[:24]:<24} "
              f"{summary.title[:44]}{flag}")
        print(f"  {'':>10}  {summary.n_number}  {summary.url}")

    if args.save:
        with RareCoinStore(full_config.rare.database) as store:
            store.init_schema()
            before = store.count()
            for entry in keepers.values():
                issue, summary = entry["issue"], entry["summary"]
                mark = f" {issue.mint_letter}" if issue.mint_letter else ""
                caveat = ("type notes errors in some years — not confirmed for "
                          "this one" if entry["type_hint"] and not entry["hint"]
                          else "")
                notes = " | ".join(part for part in (
                    f"{entry['quality']} issue", entry["note"], caveat,
                    issue.comment, entry["comments"][:300]
                ) if part)
                store.add_rare_coin(
                    country=summary.issuer,
                    denomination=entry["cents"],
                    year=issue.year,
                    variant=entry["variant"],
                    name=f"{summary.title}{mark} ({summary.n_number})"[:120],
                    mintage=issue.mintage,
                    notes=notes[:500],
                    source_url=summary.url,
                )
            print(f"\n  saved {store.count() - before} new row(s) to "
                  f"{full_config.rare.database} ({store.count()} total)")
    else:
        print("\n  Nothing written. Re-run with --save to store these.")
    return 0


def _read_targets(path: Path) -> list[dict]:
    """Read a target CSV, skipping `#` comments and blank lines."""
    if not path.exists():
        raise FileNotFoundError(f"target list not found: {path}")
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    rows = []
    for raw in csv.DictReader(lines):
        query = (raw.get("query") or "").strip()
        if not query:
            continue
        denomination = (raw.get("denomination") or "").strip()
        rows.append({
            "denomination": int(denomination) if denomination else None,
            "issuer": (raw.get("issuer") or "").strip(),
            "query": query,
            "note": (raw.get("note") or "").strip(),
        })
    return rows


def _denomination_from_title(title: str) -> int:
    """Face value in cents, read from a Numista title like '2 Euros (...)'.

    Zero when it cannot be read, rather than a guess — the column is NOT NULL
    and a wrong denomination silently excludes the coin from the rare stage's
    watchlist, which is worse than an obviously bogus 0.
    """
    import re

    lowered = title.lower()
    # Cents first: Numista writes "50 Euro Cents", so a euro-first pattern reads
    # the 50 as fifty euros.
    cents = re.match(r"\s*(\d+)\s*(?:euro\s*)?cents?\b", lowered)
    if cents:
        return int(cents.group(1))
    euros = re.match(r"\s*(\d+)\s*euros?\b", lowered)
    if euros:
        return int(euros.group(1)) * 100
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
