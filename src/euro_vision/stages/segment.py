"""Stage 1 — locate individual coins in a tray image.

Backends:
  hough     — classical circle detection. Fine for well-spaced coins, but it
              breaks down on a densely poured tray: it double-detects single
              coins and straddles touching pairs.
  watershed — threshold the coins against the tray, then split touching ones on
              the distance transform. This is the standard treatment for
              touching round objects and suits a tray of bright coins on a matte
              dark surface. Default for real trays.
  yolo      — trained detector, for when tray images have been labelled.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import SegmentConfig
from ..types import Coin, Detection, ScanResult
from .base import Stage


class SegmentStage(Stage):
    name = "segment"

    def __init__(self, config: SegmentConfig):
        self.config = config
        self._model = None  # lazily loaded YOLO model

    def describe(self) -> str:
        return f"segment[{self.config.backend}]"

    def run(self, result: ScanResult) -> ScanResult:
        image = result.meta.get("image")
        if image is None:
            raise ValueError("segment stage requires result.meta['image']")

        if self.config.backend == "hough":
            detections = self._detect_hough(image)
        elif self.config.backend == "watershed":
            detections = self._detect_watershed(image)
        elif self.config.backend == "yolo":
            detections = self._detect_yolo(image)
        else:
            raise ValueError(f"unknown segment backend: {self.config.backend}")

        # Watershed already measures from the coin's own region, and re-running
        # a local threshold would merge touching coins back together.
        if self.config.refine_edges and self.config.backend == "hough":
            detections = [
                _refine(image, d, self.config.min_radius, self.config.max_radius)
                for d in detections
            ]

        detections.sort(key=lambda d: (d.y, d.x))  # reading order, stable indices
        result.coins = [
            Coin(index=i, detection=d, image=self._crop(image, d))
            for i, d in enumerate(detections)
        ]
        return result

    # -- backends ---------------------------------------------------------

    def _detect_hough(self, image: np.ndarray) -> list[Detection]:
        cfg = self.config
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=cfg.min_distance,
            param1=100,  # Canny upper threshold
            param2=30,  # accumulator threshold; lower finds more (and more junk)
            minRadius=cfg.min_radius,
            maxRadius=cfg.max_radius,
        )
        if circles is None:
            return []

        return [
            Detection(x=int(x), y=int(y), radius=int(r), confidence=1.0)
            for x, y, r in np.round(circles[0]).astype(int)
        ]

    def _detect_watershed(self, image: np.ndarray) -> list[Detection]:
        """Split touching coins on the distance transform.

        Coins are bright and the tray is dark, so one threshold separates metal
        from tray. That leaves touching coins fused into one blob, which the
        distance transform undoes: every coin centre is a local maximum roughly
        its own radius high, while the join between two coins is a narrow neck
        with a low value. Seeding watershed above that neck therefore yields one
        region per coin.

        The radius comes from the distance transform peak rather than the
        region's area, because watershed cuts the shared boundary between
        touching coins and so understates their area.
        """
        cfg = self.config
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if cfg.coin_threshold is None:
            # Otsu assumes two balanced classes. A full tray is ~70% metal, and
            # coin faces span a wide range of brightness, so Otsu cuts too high
            # and drops the darker coppers. Measured on a real tray: Otsu chose
            # 100 and found 56 of ~80 coins; 75 found 70.
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(gray, cfg.coin_threshold, 255, cv2.THRESH_BINARY)

        # Brightness alone loses worn copper coins, which are as dark as the
        # tray. They are still strongly coloured, though, and the tray is not:
        # a saturation test recovers them without letting the tray back in.
        if cfg.copper_saturation is not None:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            copper = cv2.inRange(
                hsv,
                np.array((0, cfg.copper_saturation, cfg.copper_min_value), np.uint8),
                np.array((25, 255, 255), np.uint8),
            )
            mask = cv2.bitwise_or(mask, copper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Fill the dark relief inside each coin by redrawing external contours
        # solid. A morphological close would do this too, but it would also
        # bridge the gap between touching coins — destroying the very necks the
        # distance transform needs in order to separate them.
        filled = np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)

        distance = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
        mask = filled

        # Seed only where the distance exceeds most of the smallest coin's
        # radius. Lower and two touching coins share a seed; higher and small
        # coins get no seed at all.
        seed_level = cfg.min_radius * cfg.watershed_seed_ratio
        _, seeds = cv2.threshold(distance, seed_level, 255, cv2.THRESH_BINARY)
        seeds = seeds.astype(np.uint8)

        count, labels = cv2.connectedComponents(seeds)
        if count < 2:
            return []

        markers = labels + 1
        markers[cv2.subtract(mask, seeds) == 255] = 0  # let watershed decide these
        markers = cv2.watershed(image, markers)

        tray_level = float(np.median(gray[mask == 0])) if (mask == 0).any() else 0.0

        detections: list[Detection] = []
        for label in range(2, count + 1):
            region = (markers == label).astype(np.uint8)
            if cv2.countNonZero(region) == 0:
                continue

            # Peak of the distance transform inside this region: its location is
            # the coin centre and its value the inscribed radius.
            _, radius, _, centre = cv2.minMaxLoc(distance, mask=region)
            if not (cfg.min_radius <= radius <= cfg.max_radius):
                continue

            profile = None
            if cfg.radial_measure:
                radius, profile = _radial_radius(
                    gray, centre[0], centre[1], radius, tray_level, cfg
                )
                if not (cfg.min_radius <= radius <= cfg.max_radius):
                    continue

            detections.append(
                Detection(
                    x=int(centre[0]),
                    y=int(centre[1]),
                    radius=int(round(radius)),
                    confidence=1.0,
                    radial_profile=None if profile is None else profile.tolist(),
                )
            )
        return detections

    def _detect_yolo(self, image: np.ndarray) -> list[Detection]:
        model = self._load_yolo()
        predictions = model.predict(image, conf=self.config.confidence, verbose=False)

        detections: list[Detection] = []
        for box in predictions[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    x=int((x1 + x2) / 2),
                    y=int((y1 + y2) / 2),
                    radius=int(min(x2 - x1, y2 - y1) / 2),
                    confidence=float(box.conf[0]),
                )
            )
        return detections

    def _load_yolo(self):
        if self._model is None:
            from ultralytics import YOLO  # optional dependency, imported on use

            self._model = YOLO(self.config.weights)
        return self._model

    # -- helpers ----------------------------------------------------------

    def _measure_only(self, image: np.ndarray, det: Detection) -> Detection:
        """Exposed for tests and diagnostics."""
        return _refine(image, det, self.config.min_radius, self.config.max_radius)

    def _crop(self, image: np.ndarray, det: Detection) -> np.ndarray:
        pad = int(det.radius * self.config.crop_padding)
        h, w = image.shape[:2]
        x1 = max(0, det.x - det.radius - pad)
        y1 = max(0, det.y - det.radius - pad)
        x2 = min(w, det.x + det.radius + pad)
        y2 = min(h, det.y + det.radius + pad)
        return image[y1:y2, x1:x2].copy()


def _radial_radius(
    gray: np.ndarray,
    cx: float,
    cy: float,
    hint: float,
    tray_level: float,
    cfg,
) -> tuple[float, "np.ndarray | None"]:
    """Measure a coin's radius by walking outward from its centre.

    Returns the median radius and the per-ray radii it came from, the latter
    being what a striking-error check needs — see `euro_vision.errors`.

    Watershed finds centres reliably but measures radius as the inscribed radius
    of the thresholded mask — and that threshold also strips the coin's dim outer
    rim, so every coin reads small. Measured on a real tray the bias was 1.58 mm,
    enough to report a 2 euro coin as a 50c.

    Separating the two jobs fixes it: keep watershed's centre, then find the edge
    along rays as the point where brightness falls halfway between the coin's own
    level and the tray's. The median across rays is taken because a coin touching
    its neighbours has some rays run straight on into them; those overshoot, and
    the median ignores them as long as under half the perimeter is blocked.
    """
    height, width = gray.shape[:2]
    limit = min(cfg.max_radius * 1.25, hint * 2.2)

    inner = gray[
        max(0, int(cy - hint * 0.4)):int(cy + hint * 0.4),
        max(0, int(cx - hint * 0.4)):int(cx + hint * 0.4),
    ]
    if inner.size == 0:
        return hint, None
    coin_level = float(np.median(inner))
    edge_level = tray_level + (coin_level - tray_level) * cfg.radial_edge_fraction
    if coin_level <= tray_level:
        return hint, None

    angles = np.linspace(0, 2 * np.pi, cfg.radial_rays, endpoint=False)
    steps = np.arange(hint * 0.5, limit, 0.5)
    xs = cx + np.cos(angles)[:, None] * steps[None, :]
    ys = cy + np.sin(angles)[:, None] * steps[None, :]

    valid = (xs >= 0) & (xs < width - 1) & (ys >= 0) & (ys < height - 1)
    samples = np.full(xs.shape, tray_level, dtype=np.float32)
    xi = np.clip(xs.astype(np.int32), 0, width - 1)
    yi = np.clip(ys.astype(np.int32), 0, height - 1)
    samples[valid] = gray[yi[valid], xi[valid]]

    # First step along each ray that falls to the tray side of the edge.
    below = samples < edge_level
    hit = below.argmax(axis=1).astype(float)
    hit[~below.any(axis=1)] = len(steps) - 1  # ray never left the metal
    radii = hint * 0.5 + hit * 0.5

    # The per-ray radii go back with the median. A ray blocked by a touching
    # neighbour overshoots and a clipped edge falls short, so the two are told
    # apart by sign — but only while the individual readings survive.
    return float(np.median(radii)), radii


def _refine(
    image: np.ndarray, det: Detection, min_radius: int, max_radius: int
) -> Detection:
    """Re-measure a detection from the coin's actual outer edge.

    Hough reports the radius of the strongest accumulator peak, which is not a
    reliable measurement: it reads a few percent small on a plain disc, and on a
    coin with strong concentric structure — the inner circle of a bimetallic
    1 or 2 euro especially — it can lock onto an interior ring and report a
    radius ~20-30% short. Either error is large next to the 1 mm gap between the
    closest Euro denominations.

    So the radius is remeasured from the thresholded coin area, which finds the
    coin/tray boundary regardless of what the interior looks like. Deriving it
    from area (r = sqrt(A / pi)) averages over the whole boundary, so it is far
    more stable than any single edge crossing.

    The result is accepted only if it lands within the physically possible coin
    size range. That is a more meaningful check than comparing against Hough's
    own estimate, since a large correction is exactly what we expect when Hough
    has locked onto an inner ring.
    """
    # Enough tray in frame for Otsu to see both classes, even when the detection
    # radius is an interior ring rather than the coin's edge.
    margin = int(det.radius * 0.60)
    h, w = image.shape[:2]
    x1 = max(0, det.x - det.radius - margin)
    y1 = max(0, det.y - det.radius - margin)
    x2 = min(w, det.x + det.radius + margin)
    y2 = min(h, det.y + det.radius + margin)

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return det

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu does not know which side is the coin. The tray occupies the ROI
    # border, so whichever class dominates the border is the background.
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:
        mask = cv2.bitwise_not(mask)

    # Close small holes from the coin's own relief, which is darker than the rim.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Count pixels rather than measuring a traced contour: cv2.contourArea walks
    # the centres of boundary pixels, which understates a disc by roughly half a
    # pixel of radius. Pixel counting has no such bias.
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        return det

    # Label 0 is background; take the largest remaining component.
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = float(stats[largest, cv2.CC_STAT_AREA])
    if area <= 0:
        return det

    radius = float(np.sqrt(area / np.pi))
    if not (min_radius <= radius <= max_radius):
        return det  # not a physically plausible coin; keep the detector's word

    cx, cy = centroids[largest]
    return Detection(
        x=int(round(x1 + cx)),
        y=int(round(y1 + cy)),
        radius=int(round(radius)),
        confidence=det.confidence,
    )
