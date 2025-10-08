import numpy as np
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from analyze import plotRewards, customizer
from src.utils.common import CT

customizer.update_font_family("Arial")
customizer.update_font_size(23)


def test_sli_plot_like_paper():
    """
    Synthetic test harness to reproduce an SLI (rpid) plot like the paper figure.
    - Two trainings (T1, T2)
    - Two groups (Ctrl vs Antenna removed)
    - Controlled exp–yoked deltas per training/group.
    """
    nb = 5  # number of time bins
    nf = 2  # flies per exp–yoked pair
    ng = 2  # groups
    ntr = 2  # trainings
    pairs_per_group = 50
    nvid = ng * pairs_per_group
    rng = np.random.default_rng()

    # Weak and noisy trends near zero
    delta_ctrl_T1 = rng.normal(0.0, 0.15, nb)
    delta_ctrl_T2 = rng.normal(0.1, 0.25, nb)
    delta_ant_T1  = rng.normal(-0.05, 0.2, nb)
    delta_ant_T2  = rng.normal(0.0, 0.3, nb)

    # Optional: add one dramatic spike to simulate a "weird" bin
    delta_ctrl_T2[rng.integers(nb)] += 0.8

    # --- Define base yoked means (shared shape) ---
    yoked_base = np.array([0.4, 0.45, 0.5, 0.55, 0.6])

    # --- Collect into structured arrays ---
    deltas = [[delta_ctrl_T1, delta_ant_T1],
              [delta_ctrl_T2, delta_ant_T2]]

    fake_data = np.zeros((nvid, ntr, nf * (nb + 1)))

    for vid in range(nvid):
        g = vid // pairs_per_group  # 0=Ctrl, 1=Antenna removed
        for tr in range(ntr):
            # Choose delta curve for this group×training
            delta_curve = deltas[tr][g]
            for f in range(nf):
                # Yoked is baseline, exp = yoked + delta
                if f == 0:
                    vals = yoked_base + delta_curve  # EXP
                else:
                    vals = yoked_base                # YOKED

                # Add Gaussian noise
                vals = vals + rng.normal(0, 0.1, size=nb)
                # Occasional small outlier
                if rng.random() < 0.05:
                    vals += rng.choice([-0.25, 0.25])
                # Append NaN at end
                vals_with_nan = np.concatenate([vals, [np.nan]])
                fake_data[vid, tr, f * (nb + 1):(f + 1) * (nb + 1)] = vals_with_nan

    # --- Dummy objects expected by plotRewards ---
    class DummyTraining:
        def __init__(self, name):
            self._name = name
        def name(self):
            return self._name
        def hasSymCtrl(self):
            return False

    class DummyVA:
        def __init__(self):
            self.flies = [0, 1]
            self.saved_auc = {}
            self.numPostBuckets, self.numNonPostBuckets = None, nb
            self.rpiNumPostBuckets, self.rpiNumNonPostBuckets = None, 0
            self.speed = [0]
            self.stopFrac = [0]
            self.ct = CT.htl

    fake_va = DummyVA()
    trns = [DummyTraining("Training 1"), DummyTraining("Training 2")]
    gis = np.repeat([0, 1], pairs_per_group)
    gls = ["Ctrl large agarose", "Antenna removed large agarose"]

    # --- Compute exp–yoked difference (the actual SLI values) ---
    fake_data_diff = fake_data[:, :, 0:(nb + 1)] - fake_data[:, :, (nb + 1):2*(nb + 1)]

    # Quick check on mean deltas (sanity print)
    mean_deltas = np.nanmean(fake_data_diff, axis=(0, 1))
    print("Approx mean SLI across all videos:", mean_deltas)

    # --- Plot using plotRewards() ---
    plotRewards(
        va=fake_va,
        tp="rpid",
        a=fake_data_diff,
        trns=trns,
        gis=gis,
        gls=gls,
        vas=[fake_va] * nvid,
    )


if __name__ == "__main__":
    test_sli_plot_like_paper()
