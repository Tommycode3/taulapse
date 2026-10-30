import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# Colours: one blue for "model/estimate", one warm accent for the contrast
# group, greys for data. Okabe-Ito derived (colour-blind safe).
BLUE, ORANGE, GREY, DARK = "#0072B2", "#D55E00", "#9a9a9a", "#333333"


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def save_json(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1, default=_default)


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.frameon": False,
    })
    return plt
