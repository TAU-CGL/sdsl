from __future__ import annotations

import os
import struct
from dataclasses import dataclass
import numpy as np
from PIL import Image


@dataclass
class PgmMap:
    """Occupancy grid loaded from a ROS2 / SLAM Toolbox .yaml + .pgm map pair.

    Attributes
    ----------
    grid : np.ndarray
        Raw pixel values, shape ``(height, width)``, dtype ``uint8``.
        Convention matches the PGM file: row 0 is the *top* of the image.
        0 = black = occupied at default thresholds; 255 = white = free.
    resolution : float
        Metres per pixel.
    origin_x, origin_y : float
        World coordinates of the **bottom-left** corner of the map (metres).
        This is ``origin[0]`` and ``origin[1]`` from the YAML.
    occupied_thresh : float
        Pixels with occupancy probability above this value are treated as
        obstacles (default 0.65, matching ROS convention).
    free_thresh : float
        Pixels below this value are treated as free (default 0.196).
    negate : bool
        When ``True`` the pixel intensity encodes occupancy directly
        (lighter = more occupied) instead of the default ROS convention
        (darker = more occupied).
    """

    grid: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    occupied_thresh: float = 0.65
    free_thresh: float = 0.196
    negate: bool = False

    @property
    def width(self) -> int:
        return self.grid.shape[1]

    @property
    def height(self) -> int:
        return self.grid.shape[0]


# ---------------------------------------------------------------------------
# PGM reader (no external dependencies)
# ---------------------------------------------------------------------------

def _read_pgm(path: str) -> np.ndarray:
    """Read a PGM file (P2 ASCII or P5 binary) into a (height, width) uint8 array."""
    with open(path, "rb") as f:
        # Magic number
        magic = f.readline().strip()
        if magic not in (b"P2", b"P5"):
            raise ValueError(
                f"{path!r} is not a PGM file (magic={magic!r}); "
                "expected P2 or P5."
            )

        # Skip comments
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()

        # Width and height (may be on one line or split across lines)
        tokens: list[int] = []
        while len(tokens) < 2:
            tokens += [int(t) for t in line.split() if t and not t.startswith(b"#")]
            if len(tokens) < 2:
                line = f.readline()
        width, height = tokens[0], tokens[1]

        # Max value
        maxval = int(f.readline().strip())

        if magic == b"P5":
            # Binary: one or two bytes per sample
            dtype = np.uint8 if maxval <= 255 else np.uint16
            nbytes = height * width * dtype().itemsize
            data = np.frombuffer(f.read(nbytes), dtype=dtype)
        else:
            # ASCII
            data = np.array(f.read().split(), dtype=np.uint16 if maxval > 255 else np.uint8)

    if data.size != height * width:
        raise ValueError(
            f"PGM data size mismatch: expected {height * width} samples, "
            f"got {data.size}."
        )

    grid = data.reshape(height, width)

    if maxval != 255:
        grid = (grid.astype(np.float32) * (255.0 / maxval)).astype(np.uint8)
    else:
        grid = grid.astype(np.uint8)

    return grid


# ---------------------------------------------------------------------------
# YAML parser (handles the simple flat key: value format used by ROS)
# ---------------------------------------------------------------------------

def _parse_ros_map_yaml(yaml_path: str) -> dict:
    """Parse a ROS map_saver YAML file without requiring PyYAML."""
    params: dict = {}
    with open(yaml_path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition(":")
            if not sep:
                continue
            params[key.strip()] = value.strip()
    return params


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_pgm_map(yaml_path: str) -> PgmMap:
    """Load a ROS2 SLAM map from a YAML metadata file.

    Parameters
    ----------
    yaml_path : str
        Path to the ``.yaml`` file produced by ``ros2 run nav2_map_server
        map_saver_cli`` or SLAM Toolbox. The image can be either PGM or PNG.

    Returns
    -------
    PgmMap
        Dataclass with the raw occupancy grid and all map metadata.

    Notes
    -----
    The image path in the YAML is resolved relative to the directory that
    contains the YAML file when it is not an absolute path.
    """
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    params = _parse_ros_map_yaml(yaml_path)

    # --- image path ---
    image_path = params["image"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(yaml_dir, image_path)

    # --- scalar metadata ---
    resolution = float(params["resolution"])

    # origin: [x, y, z]  — z is meaningless for a 2D map
    origin_raw = params["origin"].strip("[] ")
    ox, oy = (float(v) for v in list(origin_raw.split(","))[:2])

    occupied_thresh = float(params.get("occupied_thresh", 0.65))
    free_thresh = float(params.get("free_thresh", 0.196))
    negate = bool(int(params.get("negate", 0)))

    # Load image based on file extension
    if image_path.lower().endswith(".png"):
        grid = _read_png(image_path)
    else:
        grid = _read_pgm(image_path)

    return PgmMap(
        grid=grid,
        resolution=resolution,
        origin_x=ox,
        origin_y=oy,
        occupied_thresh=occupied_thresh,
        free_thresh=free_thresh,
        negate=negate,
    )


def _read_png(path: str) -> np.ndarray:
    """Read a PNG file into a (height, width) uint8 array."""
    img = Image.open(path)

    if img.mode == "RGBA":
        img = img.convert("L")
    elif img.mode != "L":
        img = img.convert("L")

    grid = np.array(img, dtype=np.uint8)
    return grid
