import matplotlib.pyplot as plt
import numpy as np
import random

from src.utils.common import writeImage
from src.plotting.plot_customizer import PlotCustomizer
from src.utils.util import meanConfInt, slugify


# ── helper, very top of file (or anywhere before plot_turn_probabilities) ─────────
def _fmt_label(name, n, *, show_n):
    """Return  'name (n=…)'  only when show_n is True."""
    return f"{name} (n={n})" if show_n else name


class TurnProbabilityByDistancePlotter:
    def __init__(self, va_instances, gls=None, opts=None, plot_customizer=None):
        self.va_instances = va_instances
        self.gls = gls
        self.opts = opts
        self.use_union_filter = getattr(self.opts, "use_union_filter", False)
        self.plot_customizer = plot_customizer or PlotCustomizer()
        self.distances = list(va_instances[0].turn_prob_by_distance.keys())
        self.timeframes = ["pre_trn", "t1_start", "t2_end", "t3_end"]
        self.timeframe_titles = {
            "pre_trn": "Pre-Training",
            "t1_start": "Training 1 Start",
            "t2_end": "Training 2 End",
            "t3_end": "Training 3 End",
        }
        if gls:
            self.results_toward = {
                dist: {group: {"exp": [], "ctrl": []} for group in gls}
                for dist in self.distances
            }
            self.results_away = {
                dist: {group: {"exp": [], "ctrl": []} for group in gls}
                for dist in self.distances
            }
            self.results_all = {
                dist: {group: {"exp": [], "ctrl": []} for group in gls}
                for dist in self.distances
            }
            self.sample_sizes = {
                group: {
                    "exp": {"toward": [], "away": [], "all": []},
                    "ctrl": {"toward": [], "away": [], "all": []},
                }
                for group in gls
            }
        else:
            self.results_toward = {
                dist: {"exp": [], "ctrl": []} for dist in self.distances
            }
            self.results_away = {
                dist: {"exp": [], "ctrl": []} for dist in self.distances
            }
            self.results_all = {  # Define results_all similarly
                dist: {"exp": [], "ctrl": []} for dist in self.distances
            }
            self.sample_sizes = {
                "exp": {"toward": [], "away": [], "all": []},
                "ctrl": {"toward": [], "away": [], "all": []},
            }

    def filter_flies(self):
        exclusion_indices_union = set()

        for dist in self.distances:
            for va in self.va_instances:
                for timeframe in range(len(self.timeframes)):
                    # Create a list of conditions to check:
                    # Toward-center and away-from-center turns for
                    # experimental and control flies
                    conditions = [
                        va.turn_prob_by_distance[dist][0][timeframe][0],
                        va.turn_prob_by_distance[dist][1][timeframe][0],
                        va.turn_prob_by_distance[dist][0][timeframe][1],
                        va.turn_prob_by_distance[dist][1][timeframe][1],
                    ]

                    # Check if any condition is NaN
                    if any(np.isnan(condition) for condition in conditions):
                        if self.gls:
                            group_idx = va.gidx
                            exclusion_indices_union.add((va, timeframe, group_idx))
                        else:
                            exclusion_indices_union.add((va, timeframe))

        # Return the set of all excluded flies
        if self.gls:
            exclude_indices_group = {
                group: set(
                    k for k in exclusion_indices_union if k[2] == self.gls.index(group)
                )
                for group in self.gls
            }
            return exclude_indices_group
        else:
            return exclusion_indices_union

    def average_turn_probabilities(self):
        if self.gls:
            self.per_tick_ns = {
                grp: {
                    "exp": {
                        d: [[] for _ in self.timeframes]
                        for d in ("toward", "away", "all")
                    },
                    "ctrl": {
                        d: [[] for _ in self.timeframes]
                        for d in ("toward", "away", "all")
                    },
                }
                for grp in self.gls
            }
        else:

            self.per_tick_ns = {
                "exp": {
                    "toward": [[] for _ in self.timeframes],
                    "away": [[] for _ in self.timeframes],
                    "all": [[] for _ in self.timeframes],
                },
                "ctrl": {
                    "toward": [[] for _ in self.timeframes],
                    "away": [[] for _ in self.timeframes],
                    "all": [[] for _ in self.timeframes],
                },
            }

        if self.use_union_filter:
            exclude_indices = self.filter_flies()
        else:
            exclude_indices = {}

        for i, dist in enumerate(self.distances):
            if self.gls:
                group_means_toward = {
                    group: {"exp": [], "ctrl": []} for group in self.gls
                }
                group_means_away = {
                    group: {"exp": [], "ctrl": []} for group in self.gls
                }
                group_cis_toward = {
                    group: {"exp": [], "ctrl": []} for group in self.gls
                }
                group_cis_away = {group: {"exp": [], "ctrl": []} for group in self.gls}
                group_means_all = {group: {"exp": [], "ctrl": []} for group in self.gls}
                group_cis_all = {group: {"exp": [], "ctrl": []} for group in self.gls}
            else:
                exp_means_toward, exp_means_away = [], []
                exp_cis_toward, exp_cis_away = [], []
                ctrl_means_toward, ctrl_means_away = [], []
                ctrl_cis_toward, ctrl_cis_away = [], []
                exp_means_all, ctrl_means_all = [], []  # New for "all"
                exp_cis_all, ctrl_cis_all = [], []  # New for "all"

            for timeframe in range(len(self.timeframes)):
                if self.gls:
                    group_values_toward = {
                        group: {"exp": [], "ctrl": []} for group in self.gls
                    }
                    group_values_away = {
                        group: {"exp": [], "ctrl": []} for group in self.gls
                    }
                    group_values_all = {
                        group: {"exp": [], "ctrl": []} for group in self.gls
                    }  # New for "all"
                else:
                    exp_values_toward, exp_values_away = [], []
                    ctrl_values_toward, ctrl_values_away = [], []
                    exp_values_all, ctrl_values_all = [], []  # New for "all"

                for va in self.va_instances:
                    if self.gls:
                        group_idx = va.gidx
                        group = self.gls[group_idx]
                        if (
                            self.use_union_filter
                            and (va, timeframe, group_idx) in exclude_indices[group]
                        ):
                            continue
                        group_values_toward[group]["exp"].append(
                            va.turn_prob_by_distance[dist][0][timeframe][0]
                        )
                        group_values_away[group]["exp"].append(
                            va.turn_prob_by_distance[dist][0][timeframe][1]
                        )
                        group_values_all[group]["exp"].append(  # Aggregate "all"
                            np.nansum(
                                [
                                    va.turn_prob_by_distance[dist][0][timeframe][0],
                                    va.turn_prob_by_distance[dist][0][timeframe][1],
                                ]
                            )
                        )
                        group_values_toward[group]["ctrl"].append(
                            va.turn_prob_by_distance[dist][1][timeframe][0]
                        )
                        group_values_away[group]["ctrl"].append(
                            va.turn_prob_by_distance[dist][1][timeframe][1]
                        )
                        group_values_all[group]["ctrl"].append(  # Aggregate "all"
                            np.nansum(
                                [
                                    va.turn_prob_by_distance[dist][1][timeframe][0],
                                    va.turn_prob_by_distance[dist][1][timeframe][1],
                                ]
                            )
                        )
                    else:
                        if self.use_union_filter and (va, timeframe) in exclude_indices:
                            continue
                        exp_values_toward.append(
                            va.turn_prob_by_distance[dist][0][timeframe][0]
                        )
                        exp_values_away.append(
                            va.turn_prob_by_distance[dist][0][timeframe][1]
                        )
                        exp_values_all.append(  # Aggregate "all"
                            np.nansum(
                                [
                                    va.turn_prob_by_distance[dist][0][timeframe][0],
                                    va.turn_prob_by_distance[dist][0][timeframe][1],
                                ]
                            )
                        )
                        ctrl_values_toward.append(
                            va.turn_prob_by_distance[dist][1][timeframe][0]
                        )
                        ctrl_values_away.append(
                            va.turn_prob_by_distance[dist][1][timeframe][1]
                        )
                        ctrl_values_all.append(  # Aggregate "all"
                            np.nansum(
                                [
                                    va.turn_prob_by_distance[dist][1][timeframe][0],
                                    va.turn_prob_by_distance[dist][1][timeframe][1],
                                ]
                            )
                        )

                # Calculate means and confidence intervals as usual
                # Do the same for the new "all" category
                if self.gls:
                    for group in self.gls:
                        for key in ["exp", "ctrl"]:
                            # Implement pair-wise NaN setting and calculate "toward", "away", and "all"
                            paired_toward = zip(
                                group_values_toward[group]["exp"],
                                group_values_toward[group]["ctrl"],
                            )
                            paired_away = zip(
                                group_values_away[group]["exp"],
                                group_values_away[group]["ctrl"],
                            )
                            paired_all = zip(  # New for "all"
                                group_values_all[group]["exp"],
                                group_values_all[group]["ctrl"],
                            )
                            for idx, (val_exp, val_ctrl) in enumerate(paired_toward):
                                if np.isnan(val_exp) or np.isnan(val_ctrl):
                                    group_values_toward[group]["exp"][idx] = np.nan
                                    group_values_toward[group]["ctrl"][idx] = np.nan
                            for idx, (val_exp, val_ctrl) in enumerate(paired_away):
                                if np.isnan(val_exp) or np.isnan(val_ctrl):
                                    group_values_away[group]["exp"][idx] = np.nan
                                    group_values_away[group]["ctrl"][idx] = np.nan

                            (
                                group_mean_toward,
                                group_low_toward,
                                group_high_toward,
                                _,
                            ) = meanConfInt(group_values_toward[group][key])
                            group_mean_away, group_low_away, group_high_away, _ = (
                                meanConfInt(group_values_away[group][key])
                            )
                            group_means_toward[group][key].append(group_mean_toward)
                            group_means_away[group][key].append(group_mean_away)
                            group_cis_toward[group][key].append(
                                (group_low_toward, group_high_toward)
                            )
                            group_cis_away[group][key].append(
                                (group_low_away, group_high_away)
                            )
                            # New for "all":
                            for idx, (val_exp, val_ctrl) in enumerate(paired_all):
                                if np.isnan(val_exp) or np.isnan(val_ctrl):
                                    group_values_all[group]["exp"][idx] = np.nan
                                    group_values_all[group]["ctrl"][idx] = np.nan

                            group_mean_all, group_low_all, group_high_all, _ = (
                                meanConfInt(group_values_all[group][key])
                            )
                            group_means_all[group][key].append(group_mean_all)
                            group_cis_all[group][key].append(
                                (group_low_all, group_high_all)
                            )
                            if i == 0:
                                self.sample_sizes[group][key]["toward"].append(
                                    np.sum(~np.isnan(group_values_toward[group][key]))
                                )
                                self.sample_sizes[group][key]["away"].append(
                                    np.sum(~np.isnan(group_values_away[group][key]))
                                )
                                self.sample_sizes[group][key]["all"].append(
                                    np.sum(~np.isnan(group_values_all[group][key]))
                                )
                else:
                    # Implement pair-wise NaN setting and calculate "toward", "away", and "all"
                    paired_toward = zip(exp_values_toward, ctrl_values_toward)
                    paired_away = zip(exp_values_away, ctrl_values_away)
                    paired_all = zip(exp_values_all, ctrl_values_all)
                    for idx, (val_exp, val_ctrl) in enumerate(paired_toward):
                        if np.isnan(val_exp) or np.isnan(val_ctrl):
                            exp_values_toward[idx] = np.nan
                            ctrl_values_toward[idx] = np.nan
                    for idx, (val_exp, val_ctrl) in enumerate(paired_away):
                        if np.isnan(val_exp) or np.isnan(val_ctrl):
                            exp_values_away[idx] = np.nan
                            ctrl_values_away[idx] = np.nan

                    exp_mean_toward, exp_low_toward, exp_high_toward, _ = meanConfInt(
                        exp_values_toward
                    )
                    exp_mean_away, exp_low_away, exp_high_away, _ = meanConfInt(
                        exp_values_away
                    )
                    ctrl_mean_toward, ctrl_low_toward, ctrl_high_toward, _ = (
                        meanConfInt(ctrl_values_toward)
                    )
                    ctrl_mean_away, ctrl_low_away, ctrl_high_away, _ = meanConfInt(
                        ctrl_values_away
                    )

                    exp_means_toward.append(exp_mean_toward)
                    exp_means_away.append(exp_mean_away)
                    exp_cis_toward.append((exp_low_toward, exp_high_toward))
                    exp_cis_away.append((exp_low_away, exp_high_away))
                    ctrl_means_toward.append(ctrl_mean_toward)
                    ctrl_means_away.append(ctrl_mean_away)
                    ctrl_cis_toward.append((ctrl_low_toward, ctrl_high_toward))
                    ctrl_cis_away.append((ctrl_low_away, ctrl_high_away))
                    if i == 0:
                        self.sample_sizes["exp"]["toward"].append(
                            np.sum(~np.isnan(exp_values_toward))
                        )
                        self.sample_sizes["exp"]["away"].append(
                            np.sum(~np.isnan(exp_values_away))
                        )
                        self.sample_sizes["exp"]["all"].append(
                            np.sum(~np.isnan(exp_values_all))
                        )

                        self.sample_sizes["ctrl"]["toward"].append(
                            np.sum(~np.isnan(ctrl_values_toward))
                        )
                        self.sample_sizes["ctrl"]["away"].append(
                            np.sum(~np.isnan(ctrl_values_away))
                        )
                        self.sample_sizes["ctrl"]["all"].append(
                            np.sum(~np.isnan(ctrl_values_all))
                        )
                    # New for "all":
                    for idx, (val_exp, val_ctrl) in enumerate(paired_all):
                        if np.isnan(val_exp) or np.isnan(val_ctrl):
                            exp_values_all[idx] = np.nan
                            ctrl_values_all[idx] = np.nan

                    exp_mean_all, exp_low_all, exp_high_all, _ = meanConfInt(
                        exp_values_all
                    )
                    ctrl_mean_all, ctrl_low_all, ctrl_high_all, _ = meanConfInt(
                        ctrl_values_all
                    )

                    exp_means_all.append(exp_mean_all)
                    exp_cis_all.append((exp_low_all, exp_high_all))
                    ctrl_means_all.append(ctrl_mean_all)
                    ctrl_cis_all.append((ctrl_low_all, ctrl_high_all))

                # ————— record per-tick sample sizes if skipping union —————
                if not self.use_union_filter:
                    if self.gls:
                        for group in self.gls:
                            # experimental
                            self.per_tick_ns[group]["exp"]["toward"][timeframe].append(
                                int(
                                    np.sum(~np.isnan(group_values_toward[group]["exp"]))
                                )
                            )
                            self.per_tick_ns[group]["exp"]["away"][timeframe].append(
                                int(np.sum(~np.isnan(group_values_away[group]["exp"])))
                            )
                            self.per_tick_ns[group]["exp"]["all"][timeframe].append(
                                int(np.sum(~np.isnan(group_values_all[group]["exp"])))
                            )
                            # control
                            self.per_tick_ns[group]["ctrl"]["toward"][timeframe].append(
                                int(
                                    np.sum(
                                        ~np.isnan(group_values_toward[group]["ctrl"])
                                    )
                                )
                            )
                            self.per_tick_ns[group]["ctrl"]["away"][timeframe].append(
                                int(np.sum(~np.isnan(group_values_away[group]["ctrl"])))
                            )
                            self.per_tick_ns[group]["ctrl"]["all"][timeframe].append(
                                int(np.sum(~np.isnan(group_values_all[group]["ctrl"])))
                            )
                    else:
                        self.per_tick_ns["exp"]["toward"][timeframe].append(
                            int(np.sum(~np.isnan(exp_values_toward)))
                        )
                        self.per_tick_ns["exp"]["away"][timeframe].append(
                            int(np.sum(~np.isnan(exp_values_away)))
                        )
                        self.per_tick_ns["exp"]["all"][timeframe].append(
                            int(np.sum(~np.isnan(exp_values_all)))
                        )
                        # control
                        self.per_tick_ns["ctrl"]["toward"][timeframe].append(
                            int(np.sum(~np.isnan(ctrl_values_toward)))
                        )
                        self.per_tick_ns["ctrl"]["away"][timeframe].append(
                            int(np.sum(~np.isnan(ctrl_values_away)))
                        )
                        self.per_tick_ns["ctrl"]["all"][timeframe].append(
                            int(np.sum(~np.isnan(ctrl_values_all)))
                        )

            # Store the results
            if self.gls:
                for group in self.gls:
                    self.results_toward[dist][group]["exp"] = {
                        "means": group_means_toward[group]["exp"],
                        "cis": group_cis_toward[group]["exp"],
                    }
                    self.results_toward[dist][group]["ctrl"] = {
                        "means": group_means_toward[group]["ctrl"],
                        "cis": group_cis_toward[group]["ctrl"],
                    }
                    self.results_away[dist][group]["exp"] = {
                        "means": group_means_away[group]["exp"],
                        "cis": group_cis_away[group]["exp"],
                    }
                    self.results_away[dist][group]["ctrl"] = {
                        "means": group_means_away[group]["ctrl"],
                        "cis": group_cis_away[group]["ctrl"],
                    }
                    self.results_all[dist][group] = {
                        "exp": {
                            "means": group_means_all[group]["exp"],
                            "cis": group_cis_all[group]["exp"],
                        },
                        "ctrl": {
                            "means": group_means_all[group]["ctrl"],
                            "cis": group_cis_all[group]["ctrl"],
                        },
                    }
            else:
                self.results_toward[dist]["exp"] = {
                    "means": exp_means_toward,
                    "cis": exp_cis_toward,
                }
                self.results_toward[dist]["ctrl"] = {
                    "means": ctrl_means_toward,
                    "cis": ctrl_cis_toward,
                }
                self.results_away[dist]["exp"] = {
                    "means": exp_means_away,
                    "cis": exp_cis_away,
                }
                self.results_away[dist]["ctrl"] = {
                    "means": ctrl_means_away,
                    "cis": ctrl_cis_away,
                }
                self.results_all[dist] = {
                    "exp": {"means": exp_means_all, "cis": exp_cis_all},
                    "ctrl": {"means": ctrl_means_all, "cis": ctrl_cis_all},
                }

    def plot_turn_probabilities(self):
        colors = {
            0: {"fill": "tomato", "edge": "darkred"},
            1: {"fill": "forestgreen", "edge": "darkgreen"},
        }

        def get_random_color():
            return "#{:06x}".format(random.randint(0, 0xFFFFFF))

        def plot_data(
            distances, means, cis, labels, title, filename, colors, legend_loc, ns=None
        ):
            figure_size = 6
            fig, ax = plt.subplots(1, 1, figsize=(figure_size, figure_size))
            # zip together each curve’s mean/ci/label/color—and its ns-list if provided
            for idx, (mean, ci, label, color) in enumerate(
                zip(means, cis, labels, colors)
            ):
                low, high = zip(*ci)
                ec, fc = color["edge"], color["fill"]
                ax.plot(distances, mean, label=label, marker="o", color=ec)
                ax.fill_between(distances, low, high, alpha=0.2, color=fc)

            dist_ref_pt = (
                "reward circle"
                if self.opts.contact_geometry == "circular"
                else "chamber"
            )
            ax.set_xlabel(
                f"Distance from {dist_ref_pt} center (mm)",
            )
            ax.set_ylabel("Turn probability")
            ax.grid(False)
            ax.set_ylim(bottom=0)

            legend_inside = True

            if legend_inside:
                legend = ax.legend(
                    loc=legend_loc, borderaxespad=0.2, prop={"style": "italic"}
                )
            else:
                box = ax.get_position()  # get the original axis bounds
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

                legend = ax.legend(
                    loc=legend_loc,
                    bbox_to_anchor=(1.02, 1),
                    borderaxespad=0.0,
                    prop={"style": "italic"},
                )

            self.plot_customizer.adjust_padding_proportionally()
            kwargs = {}
            if self.opts:
                kwargs["format"] = self.opts.imageFormat
            writeImage(filename, **kwargs)
            plt.close()

        for direction, results in [
            ("toward", self.results_toward),
            ("away", self.results_away),
            ("all", self.results_all),
        ]:
            legend_loc = (
                "upper left" if direction in ("toward", "all") else "upper right"
            )
            for i in range(len(self.timeframes)):
                # Plot experimental vs. experimental across groups
                if self.gls:
                    means = [
                        [
                            results[dist][group]["exp"]["means"][i]
                            for dist in self.distances
                        ]
                        for group in self.gls
                    ]
                    cis = [
                        [
                            results[dist][group]["exp"]["cis"][i]
                            for dist in self.distances
                        ]
                        for group in self.gls
                    ]
                    show_n = self.use_union_filter
                    labels = [
                        _fmt_label(
                            group,
                            self.sample_sizes[group]["exp"][direction][i],
                            show_n=show_n,
                        )
                        for group in self.gls
                    ]
                    group_colors = [
                        (
                            {"fill": get_random_color(), "edge": get_random_color()}
                            if idx > 2
                            else colors[idx]
                        )
                        for idx in range(len(self.gls))
                    ]
                    extra = {}
                    if not self.use_union_filter:
                        extra["ns"] = [
                            self.per_tick_ns[group]["exp"][direction][i]
                            for group in self.gls
                        ]
                    plot_data(
                        self.distances,
                        means,
                        cis,
                        labels,
                        f"Turn Probability by Distance ({direction.capitalize()}) - {self.timeframe_titles[self.timeframes[i]]}",
                        f"imgs/turn_probability_{self.timeframes[i]}_{direction}_exp_across_groups.png",
                        group_colors,
                        legend_loc,
                        **extra,
                    )

                    # Plot experimental vs. yoked control within each group
                    for group in self.gls:
                        means = [
                            [
                                results[dist][group][key]["means"][i]
                                for dist in self.distances
                            ]
                            for key in ["exp", "ctrl"]
                        ]
                        cis = [
                            [
                                results[dist][group][key]["cis"][i]
                                for dist in self.distances
                            ]
                            for key in ["exp", "ctrl"]
                        ]
                        show_n_exp = (
                            self.use_union_filter
                        )  # experimental label gets n only w/ union filter
                        show_n_ctrl = False  # yoked never shows n (per your spec)

                        labels = [
                            _fmt_label(
                                "Experimental",
                                self.sample_sizes[group]["exp"][direction][i],
                                show_n=show_n_exp,
                            ),
                            _fmt_label(
                                "Yoked",  # yoked sample size omitted
                                self.sample_sizes[group]["ctrl"][direction][i],
                                show_n=show_n_ctrl,
                            ),
                        ]
                        extra = {}
                        if not self.use_union_filter:
                            extra["ns"] = [
                                self.per_tick_ns[group]["exp"][direction][i],
                                self.per_tick_ns[group]["ctrl"][direction][i],
                            ]
                        plot_data(
                            self.distances,
                            means,
                            cis,
                            labels,
                            f"Turn Probability by Distance ({direction.capitalize()}) - {self.timeframe_titles[self.timeframes[i]]}\n{group}",
                            f"imgs/turn_probability_{self.timeframes[i]}_{direction}_{slugify(group)}.png",
                            [
                                {"fill": "#1f4da1", "edge": "#1f4da1"},
                                {"fill": "#a00000", "edge": "#a00000"},
                            ],
                            legend_loc,
                            **extra,
                        )

                else:
                    means = [
                        [results[dist]["exp"]["means"][i] for dist in self.distances],
                        [results[dist]["ctrl"]["means"][i] for dist in self.distances],
                    ]
                    cis = [
                        [results[dist]["exp"]["cis"][i] for dist in self.distances],
                        [results[dist]["ctrl"]["cis"][i] for dist in self.distances],
                    ]
                    show_n_exp = self.use_union_filter
                    show_n_ctrl = False
                    labels = [
                        _fmt_label(
                            "Experimental",
                            self.sample_sizes["exp"][direction][i],
                            show_n=show_n_exp,
                        ),
                        _fmt_label(
                            "Yoked",
                            self.sample_sizes["ctrl"][direction][i],
                            show_n=show_n_ctrl,
                        ),
                    ]
                    extra = {}
                    if not self.use_union_filter:
                        extra["ns"] = [
                            self.per_tick_ns["exp"][direction][i],
                            self.per_tick_ns["ctrl"][direction][i],
                        ]
                    plot_data(
                        self.distances,
                        means,
                        cis,
                        labels,
                        f"Turn Probability by Distance ({direction.capitalize()}) - {self.timeframe_titles[self.timeframes[i]]}",
                        f"imgs/turn_probability_{self.timeframes[i]}_{direction}.png",
                        [
                            {"fill": "#1f4da1", "edge": "#1f4da1"},
                            {"fill": "#a00000", "edge": "#a00000"},
                        ],
                        legend_loc,
                        **extra,
                    )

            # Plot Training 1 Start vs. Training 3 End for the same category
            if self.gls:
                for group in self.gls:
                    for key in ["exp", "ctrl"]:
                        t1_means = [
                            results[dist][group][key]["means"][
                                self.timeframes.index("t1_start")
                            ]
                            for dist in self.distances
                        ]
                        t1_cis = [
                            results[dist][group][key]["cis"][
                                self.timeframes.index("t1_start")
                            ]
                            for dist in self.distances
                        ]
                        t3_means = [
                            results[dist][group][key]["means"][
                                self.timeframes.index("t3_end")
                            ]
                            for dist in self.distances
                        ]
                        t3_cis = [
                            results[dist][group][key]["cis"][
                                self.timeframes.index("t3_end")
                            ]
                            for dist in self.distances
                        ]

                        means = [t1_means, t3_means]
                        cis = [t1_cis, t3_cis]
                        timeframe_indices = [
                            self.timeframes.index("t1_start"),
                            self.timeframes.index("t3_end"),
                        ]
                        show_n = (
                            self.use_union_filter
                        )  # only when union filter is active
                        labels = [
                            _fmt_label(
                                "Training 1 Start",
                                self.sample_sizes[group][key][direction][
                                    timeframe_indices[0]
                                ],
                                show_n=show_n,
                            ),
                            _fmt_label(
                                "Training 3 End",
                                self.sample_sizes[group][key][direction][
                                    timeframe_indices[1]
                                ],
                                show_n=show_n,
                            ),
                        ]
                        extra = {}
                        if not self.use_union_filter:
                            extra["ns"] = [
                                self.per_tick_ns[group][key][direction][index]
                                for index in timeframe_indices
                            ]
                        plot_data(
                            self.distances,
                            means,
                            cis,
                            labels,
                            f"Turn Probability by Distance ({direction.capitalize()}) - Training 1 Start vs. Training 3 End\n{group} - {key.capitalize()}",
                            f"imgs/turn_probability_t1_vs_t3_{direction}_{slugify(group)}_{key}.png",
                            [
                                {"fill": "#1f4da1", "edge": "#1f4da1"},
                                {"fill": "#a00000", "edge": "#a00000"},
                            ],
                            legend_loc,
                            **extra,
                        )
            else:
                t1_means_exp = [
                    results[dist]["exp"]["means"][self.timeframes.index("t1_start")]
                    for dist in self.distances
                ]
                t1_cis_exp = [
                    results[dist]["exp"]["cis"][self.timeframes.index("t1_start")]
                    for dist in self.distances
                ]
                t3_means_exp = [
                    results[dist]["exp"]["means"][self.timeframes.index("t3_end")]
                    for dist in self.distances
                ]
                t3_cis_exp = [
                    results[dist]["exp"]["cis"][self.timeframes.index("t3_end")]
                    for dist in self.distances
                ]

                t1_means_ctrl = [
                    results[dist]["ctrl"]["means"][self.timeframes.index("t1_start")]
                    for dist in self.distances
                ]
                t1_cis_ctrl = [
                    results[dist]["ctrl"]["cis"][self.timeframes.index("t1_start")]
                    for dist in self.distances
                ]
                t3_means_ctrl = [
                    results[dist]["ctrl"]["means"][self.timeframes.index("t3_end")]
                    for dist in self.distances
                ]
                t3_cis_ctrl = [
                    results[dist]["ctrl"]["cis"][self.timeframes.index("t3_end")]
                    for dist in self.distances
                ]

                means_exp = [t1_means_exp, t3_means_exp]
                cis_exp = [t1_cis_exp, t3_cis_exp]
                timeframe_indices = [
                    self.timeframes.index("t1_start"),
                    self.timeframes.index("t3_end"),
                ]
                show_n = self.use_union_filter  # only when union filter is active
                labels = [
                    _fmt_label(
                        "Training 1 Start",
                        self.sample_sizes["exp"][direction][timeframe_indices[0]],
                        show_n=show_n,
                    ),
                    _fmt_label(
                        "Training 3 End",
                        self.sample_sizes["exp"][direction][timeframe_indices[1]],
                        show_n=show_n,
                    ),
                ]
                extra = {}
                if not self.use_union_filter:
                    extra["ns"] = [
                        self.per_tick_ns["exp"][direction][index]
                        for index in timeframe_indices
                    ]
                plot_data(
                    self.distances,
                    means_exp,
                    cis_exp,
                    labels,
                    f"Turn Probability by Distance ({direction.capitalize()}) - Training 1 Start vs. Training 3 End\nSingle Group Analysis - Experimental",
                    f"imgs/turn_probability_t1_vs_t3_{direction}_single_group_exp.png",
                    [
                        {"fill": "#1f4da1", "edge": "#1f4da1"},
                        {"fill": "#a00000", "edge": "#a00000"},
                    ],
                    legend_loc,
                    **extra,
                )

                means_ctrl = [t1_means_ctrl, t3_means_ctrl]
                cis_ctrl = [t1_cis_ctrl, t3_cis_ctrl]
                timeframe_indices = [
                    self.timeframes.index("t1_start"),
                    self.timeframes.index("t3_end"),
                ]
                show_n = self.use_union_filter
                labels_ctrl = [
                    _fmt_label(
                        "Training 1 Start",
                        self.sample_sizes["ctrl"][direction][timeframe_indices[0]],
                        show_n=show_n,
                    ),
                    _fmt_label(
                        "Training 3 End",
                        self.sample_sizes["ctrl"][direction][timeframe_indices[1]],
                        show_n=show_n,
                    ),
                ]
                extra = {}
                if not self.use_union_filter:
                    extra["ns"] = [
                        self.per_tick_ns["ctrl"][direction][index]
                        for index in timeframe_indices
                    ]
                plot_data(
                    self.distances,
                    means_ctrl,
                    cis_ctrl,
                    labels_ctrl,
                    f"Turn Probability by Distance ({direction.capitalize()}) - Training 1 Start vs. Training 3 End\nSingle Group Analysis - Yoked",
                    f"imgs/turn_probability_t1_vs_t3_{direction}_single_group_ctrl.png",
                    [
                        {"fill": "#1f4da1", "edge": "#1f4da1"},
                        {"fill": "#a00000", "edge": "#a00000"},
                    ],
                    legend_loc,
                    **extra,
                )
