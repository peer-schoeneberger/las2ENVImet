# roots.py
# Root zone model for ENVI-met 3D plant
import math
from typing import Dict, List, Optional, Sequence

LEVELS = 10

SHAPE_CYLINDER = 'cylinder'
SHAPE_ELLIPSOID = 'ellipsoid'      # Half sphere / half ellipsoid, widest at the surface
SHAPE_HEART = 'heart'
SHAPE_TAP = 'tap'                  # Tap root: wide shallow collar, narrow deep core
SHAPE_PLATE = 'plate'              # Flat / plate root system: wide and shallow
SHAPE_CUSTOM = 'custom'            # User-defined radius profile
SHAPES = (SHAPE_CYLINDER, SHAPE_ELLIPSOID, SHAPE_HEART, SHAPE_TAP, SHAPE_PLATE, SHAPE_CUSTOM)

DIST_UNIFORM = 'uniform'
DIST_LINEAR = 'linear'             # Decreasing linearly to zero at the bottom
DIST_EXPONENTIAL = 'exponential'   # Gale & Grigal (1987), d in cm
DIST_LOGISTIC = 'logistic'         # Schenk & Jackson (2002): r(D) = 1 / (1 + (D / D50)^c)
DIST_CUSTOM = 'custom'
DISTRIBUTIONS = (DIST_UNIFORM, DIST_LINEAR, DIST_EXPONENTIAL, DIST_LOGISTIC, DIST_CUSTOM)

DEFAULT_BETA = 0.966               # Temperate deciduous forest, Jackson et al. (1996), Table 1
DEFAULT_D50 = 21.0                 # cm, Schenk & Jackson (2002), Table 4
DEFAULT_D95 = 104.0                # cm


def logistic_shape(d50: float, d95: float) -> float:
    # Shape parameter c of the logistic dose-response curve from the 50 % and 95 % rooting depths
    if d50 <= 0 or d95 <= d50:
        raise ValueError("D95 must be greater than D50, and both greater than zero.")
    return math.log(1.0 / 19.0) / math.log(d95 / d50)


def logistic_cumulative(depth_cm: float, d50: float, d95: float) -> float:
    # Cumulative root fraction above depth_cm (Schenk & Jackson 2002, Eq. 1 with Rmax = 1).
    if depth_cm <= 0:
        return 0.0
    return 1.0 / (1.0 + (depth_cm / d50) ** logistic_shape(d50, d95))


def _mid_depths():
    # Relative depth (0 = surface, 1 = bottom) of the middle of each layer.
    return [(n + 0.5) / LEVELS for n in range(LEVELS)]


def shape_profile(shape: str, custom: Optional[Sequence[float]] = None) -> List[float]:
    # Relative radius (0..1) of the ten layers for a geometric shape. Purely geometric forms.
    t = _mid_depths()
    if shape == SHAPE_CYLINDER or shape == SHAPE_PLATE:
        # A plate root system is a cylinder that is simply shallow and wide (set by depth/diameter)
        profile = [1.0] * LEVELS
    elif shape == SHAPE_ELLIPSOID:
        profile = [math.sqrt(max(0.0, 1.0 - x * x)) for x in t]
    elif shape == SHAPE_HEART:
        # Ellipsoid whose widest section lies at 25 % of the depth
        centre = 0.25
        profile = [math.sqrt(max(0.0, 1.0 - ((x - centre) / (centre if x < centre else 1.0 - centre)) ** 2 *
                                 (0.36 if x < centre else 1.0))) for x in t]
    elif shape == SHAPE_TAP:
        # Lateral roots in the upper 20 %, below that a core of 25 % of the radius tapering to 10 %
        profile = [1.0 if x <= 0.2 else 0.25 - 0.15 * (x - 0.2) / 0.8 for x in t]
    elif shape == SHAPE_CUSTOM:
        if custom is None or len(custom) != LEVELS:
            raise ValueError("A custom root shape needs exactly %d radius values." % LEVELS)
        profile = [float(v) for v in custom]
    else:
        raise ValueError("Unknown root shape: %r" % (shape,))
    return [min(1.0, max(0.0, round(v, 4))) for v in profile]


def distribution_profile(distribution: str, depth: float, beta: float = DEFAULT_BETA,
                         custom: Optional[Sequence[float]] = None,
                         d50: float = DEFAULT_D50, d95: float = DEFAULT_D95) -> List[float]:
    # Relative root density of the ten layers (mean of the returned values is 1).
    if distribution == DIST_UNIFORM:
        weights = [1.0] * LEVELS
    elif distribution == DIST_LINEAR:
        weights = [1.0 - x for x in _mid_depths()]
    elif distribution == DIST_EXPONENTIAL:
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be between 0 and 1.")
        step_cm = depth * 100.0 / LEVELS
        weights = [beta ** (n * step_cm) - beta ** ((n + 1) * step_cm) for n in range(LEVELS)]
    elif distribution == DIST_LOGISTIC:
        step_cm = depth * 100.0 / LEVELS
        weights = [logistic_cumulative((n + 1) * step_cm, d50, d95) - logistic_cumulative(n * step_cm, d50, d95)
                   for n in range(LEVELS)]
    elif distribution == DIST_CUSTOM:
        if custom is None or len(custom) != LEVELS:
            raise ValueError("A custom depth profile needs exactly %d values." % LEVELS)
        weights = [max(0.0, float(v)) for v in custom]
    else:
        raise ValueError("Unknown vertical distribution: %r" % (distribution,))
    mean = sum(weights) / LEVELS
    if mean <= 0:
        raise ValueError("The vertical root distribution is empty.")
    return [w / mean for w in weights]


class RootSystem:
    # Root zone in ENVI-met terms. All exporters write exactly these four items.

    def __init__(self, depth: float, diameter: float, rad: Sequence[float], root_range: Sequence[float],
                 settings: Optional[Dict] = None):
        if depth <= 0 or diameter <= 0:
            raise ValueError("Root depth and root diameter must be greater than zero.")
        if len(rad) != LEVELS or len(root_range) != LEVELS:
            raise ValueError("RAD and root range need exactly %d values each." % LEVELS)
        if any(v < 0 for v in rad) or any(not 0.0 <= v <= 1.0 for v in root_range):
            raise ValueError("RAD must be >= 0 and the root range between 0 and 1.")
        self.depth = float(depth)
        self.diameter = float(diameter)
        self.rad = [float(v) for v in rad]
        self.root_range = [float(v) for v in root_range]
        self.settings = dict(settings or {})     # How it was generated (for project files / reports)

    # Geometry of one layer
    @property
    def layer_thickness(self):
        return self.depth / LEVELS

    def layer_radii(self):
        return [r * self.diameter / 2.0 for r in self.root_range]

    def layer_volumes(self):
        return [math.pi * r * r * self.layer_thickness for r in self.layer_radii()]

    def total_root_area(self):
        # Sum over the layers of RAD * layer volume [m2].
        return sum(d * v for d, v in zip(self.rad, self.layer_volumes()))

    def volume(self):
        return sum(self.layer_volumes())

    def mean_rad(self):
        vol = self.volume()
        return self.total_root_area() / vol if vol > 0 else 0.0

    def to_dict(self):
        return {'depth': self.depth, 'diameter': self.diameter, 'rad': list(self.rad),
                'root_range': list(self.root_range), 'settings': dict(self.settings)}

    @classmethod
    def from_dict(cls, data):
        return cls(data['depth'], data['diameter'], data['rad'], data['root_range'], data.get('settings'))


def build_root_system(shape: str = SHAPE_CYLINDER, depth: float = 2.0, diameter: Optional[float] = None,
                      crown_diameter: Optional[float] = None, radius_factor: float = 1.0,
                      distribution: str = DIST_UNIFORM, beta: float = DEFAULT_BETA,
                      rad: Optional[float] = 0.1, total_root_area: Optional[float] = None,
                      custom_range: Optional[Sequence[float]] = None,
                      custom_depth_profile: Optional[Sequence[float]] = None,
                      d50: float = DEFAULT_D50, d95: float = DEFAULT_D95) -> RootSystem:
    # Create a RootSystem from user settings
    if diameter is None:
        if crown_diameter is None:
            raise ValueError("Give either the root diameter or the crown diameter and a factor.")
        diameter = crown_diameter * radius_factor
    root_range = shape_profile(shape, custom_range)
    weights = distribution_profile(distribution, depth, beta, custom_depth_profile, d50, d95)
    # Layers without horizontal extent cannot hold roots
    weights = [w if r > 0 else 0.0 for w, r in zip(weights, root_range)]
    unit = RootSystem(depth, diameter, weights, root_range)
    if total_root_area is not None:
        if total_root_area < 0:
            raise ValueError("The total root area must not be negative.")
        base = unit.total_root_area()
        factor = total_root_area / base if base > 0 else 0.0
    else:
        if rad is None or rad < 0:
            raise ValueError("The root area density must not be negative.")
        base = unit.mean_rad()
        factor = rad / base if base > 0 else 0.0
    values = [round(w * factor, 5) for w in weights]
    settings = {'shape': shape, 'depth': depth, 'diameter': diameter, 'crown_diameter': crown_diameter,
                'radius_factor': radius_factor, 'distribution': distribution, 'beta': beta, 'd50': d50, 'd95': d95, 'rad': rad,
                'total_root_area': total_root_area,
                'custom_range': list(custom_range) if custom_range is not None else None,
                'custom_depth_profile': list(custom_depth_profile) if custom_depth_profile is not None else None}
    return RootSystem(depth, diameter, values, root_range, settings)


def side_view_outline(root: RootSystem):
    # Polyline (x, z) of the right half of the root zone for previews; z is negative downwards.
    points = [(0.0, 0.0)]
    step = root.layer_thickness
    for n, radius in enumerate(root.layer_radii()):
        points.append((radius, -n * step))
        points.append((radius, -(n + 1) * step))
    points.append((0.0, -root.depth))
    return points
