import numpy as np

import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from analyze import plotRewards, customizer
from src.utils.common import CT

customizer.update_font_family("Arial")
customizer.update_font_size(25)


def test_exp_vs_yoked_two_groups():
    """
    Contrived test harness for plotRewards with 2 trainings,
    2 groups, and ~70 exp-yoked pairs per group.
    Experimental (f=0) is ~0.2-0.4 higher than yoked (f=1),
    with added variation and occasional outliers.
    """
    nb = 5  # sync buckets
    nf = 2  # flies (exp, yoked)
    ng = 2  # groups
    ntr = 2  # trainings
    pairs_per_group = 70
    nvid = ng * pairs_per_group  # one "video" = one exp–yoked pair

    # --- Base pattern for exp vs yoked ---
    yoked_base = np.array([0.3, 0.35, 0.25, 0.32, 0.28])
    exp_base = yoked_base + np.array([0.0, 0.0, 0.0, 0.0, 0.0]) + 1.2

    # --- Build fake data array ---
    fake_data = np.zeros((nvid, ntr, nf * (nb + 1)))

    rng = np.random.default_rng()  # reproducible
    for vid in range(nvid):
        g = vid // pairs_per_group
        for tr in range(ntr):
            for f in range(nf):
                base_vals = exp_base if f == 0 else yoked_base
                # systematic offsets by training & group
                vals = base_vals + 0.05 * tr + 0.02 * g
                # add Gaussian noise
                vals = vals + rng.normal(0, 0.05, size=nb)
                # occasional outliers (1 in 20 flies)
                if rng.random() < 0.05:
                    vals = vals + rng.choice([-0.5, 0.5])

                # append with NaN bucket at the end
                vals_with_nan = np.concatenate([vals, [np.nan]])
                fake_data[vid, tr, f * (nb + 1) : (f + 1) * (nb + 1)] = vals_with_nan

    # --- Dummy objects for plotRewards ---
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
    trns = [DummyTraining("T1"), DummyTraining("T2")]
    gis = np.repeat([0, 1], pairs_per_group)  # group index for each pair
    gls = ["Group A", "Group B"]

    # --- Call plotRewards ---
    print("fake data shape:", fake_data.shape)
    plotRewards(
        va=fake_va,
        tp="rpi",
        a=fake_data,
        trns=trns,
        gis=gis,
        gls=gls,
        vas=[fake_va] * nvid,
        manual_overrides=None,
    )

    fake_data_diff = (
        fake_data[:, :, 0 : (nb + 1)] - fake_data[:, :, (nb + 1) : 2 * (nb + 1)]
    )

    plotRewards(
        va=fake_va,
        tp="rpid",
        a=fake_data_diff,
        trns=trns,
        gis=gis,
        gls=gls,
        vas=[fake_va] * nvid,
        manual_overrides=None,
    )


if __name__ == "__main__":
    test_exp_vs_yoked_two_groups()
