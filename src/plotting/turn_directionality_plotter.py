import os
import matplotlib.pyplot as plt

from src.utils.common import writeImage
from src.plotting.plot_customizer import PlotCustomizer
from src.utils.util import meanConfInt, slugify


class TurnDirectionalityPlotter:
    def __init__(self, vas, gls=None, plot_customizer=None, opts=None):
        self.vas = vas
        self.gls = gls
        self.plot_customizer = plot_customizer or PlotCustomizer()
        self.opts = opts

    def extract_turn_data(self):
        per_va_turn_data = {}
        for va in self.vas:
            if any([trj.bad() for trj in va.trx]):
                continue
            for training in va.trns:
                turn_data = {}
                for idx, trx in enumerate(va.trx):
                    if trx.bad():
                        continue
                    fly_type = "experimental" if idx == 0 else "yoked control"
                    for boundary_type, orientations in trx.boundary_event_stats.items():
                        for orientation, ref_points in orientations.items():
                            for ref_point, events in ref_points.items():
                                turn_direction_data = events.get(
                                    "turn_direction_toward_ctr", {}
                                )
                                for (
                                    turn_type,
                                    frame_data,
                                ) in turn_direction_data.items():
                                    frame_data_filtered = {
                                        frame: data
                                        for frame, data in frame_data.items()
                                        if training.start <= frame <= training.stop
                                    }
                                    key = (
                                        training.n,
                                        boundary_type,
                                        orientation,
                                        ref_point,
                                        turn_type,
                                        fly_type,
                                        va.gidx,  # Add group index to the key
                                    )
                                    if key not in turn_data:
                                        turn_data[key] = []
                                    turn_data[key].extend(frame_data_filtered.items())
                per_va_turn_data[(id(va), training.n)] = turn_data
        return per_va_turn_data

    def plot_turn_directionality(self):
        per_va_turn_data = self.extract_turn_data()
        all_training_proportions = {}

        for (va_id, training_n), turn_data in per_va_turn_data.items():
            for key, frames in turn_data.items():
                toward_count = sum(1 for _, toward in frames if toward)
                away_count = sum(1 for _, toward in frames if not toward)
                total = toward_count + away_count
                proportion = toward_count / total if total > 0 else 0
                if training_n not in all_training_proportions:
                    all_training_proportions[training_n] = {}
                if key not in all_training_proportions[training_n]:
                    all_training_proportions[training_n][key] = []
                all_training_proportions[training_n][key].append(proportion)

        def plot_pie_chart(sizes, labels, title, filename):
            figure_size = 6 * self.plot_customizer.increase_factor
            plt.figure(figsize=(figure_size, figure_size))

            def custom_autopct(pct):
                return f"{pct:.1f}%" if pct > 0 else ""

            plt.pie(
                sizes,
                labels=labels,
                autopct=custom_autopct,
                textprops={"fontsize": self.plot_customizer.in_plot_font_size},
                explode=(0.1, 0.1),
            )
            plt.title(title, fontsize=self.plot_customizer.font_size)
            self.plot_customizer.adjust_padding_proportionally()
            plt.tight_layout()
            kwargs = {}
            if self.opts:
                kwargs["format"] = self.opts.imageFormat
            writeImage(os.path.join("imgs", filename), **kwargs)
            plt.close()

        if self.gls:
            group_training_proportions = {group: {} for group in self.gls}

            for training, data in all_training_proportions.items():
                for key, proportions in data.items():
                    group_label = self.gls[key[-1]]
                    if training not in group_training_proportions[group_label]:
                        group_training_proportions[group_label][training] = {}
                    if key not in group_training_proportions[group_label][training]:
                        group_training_proportions[group_label][training][key] = []
                    group_training_proportions[group_label][training][key].extend(
                        proportions
                    )

            for group, training_data in group_training_proportions.items():
                for training, data in training_data.items():
                    for key, proportions in data.items():
                        mean, lower, upper, num_samples = meanConfInt(proportions)
                        mean_away, lower_away, upper_away, _ = meanConfInt(
                            [1 - p for p in proportions]
                        )
                        sizes = [mean, 1 - mean]
                        labels = [
                            f"Toward Center\nCI: [{lower*100:.1f}%, {upper*100:.1f}%]",
                            f"Away from Center\nCI: [{lower_away*100:.1f}%, {upper_away*100:.1f}%]",
                        ]
                        (
                            boundary_type,
                            orientation,
                            ref_point,
                            turn_type,
                            fly_type,
                            gidx,
                        ) = key[1:]
                        if ref_point == "ctr":
                            ref_pt_desc = "body center"
                        elif ref_point == "edge":
                            ref_pt_desc = "edge of ellipse"
                        if turn_type == "all":
                            turn_type_desc = ""
                        else:
                            turn_type_desc = f", {turn_type}"
                        title_str = (
                            f"Training {training}, {fly_type} fly, sharp turns, "
                            f"{ref_pt_desc}{turn_type_desc}\n{group} (n = {num_samples})"
                        )
                        filename = (
                            f"turn_directionality_t{training}_{fly_type}_"
                            f"{boundary_type}_{orientation}_{ref_point}_"
                            f"{turn_type}_{slugify(group)}.png"
                        )
                        plot_pie_chart(sizes, labels, title_str, filename)
        else:
            for training, data in all_training_proportions.items():
                for key, proportions in data.items():
                    mean, lower, upper, num_samples = meanConfInt(proportions)
                    mean_away, lower_away, upper_away, _ = meanConfInt(
                        [1 - p for p in proportions]
                    )
                    sizes = [mean, 1 - mean]
                    labels = [
                        f"Toward Center\nCI: [{lower*100:.1f}%, {upper*100:.1f}%]",
                        f"Away from Center\nCI: [{lower_away*100:.1f}%, {upper_away*100:.1f}%]",
                    ]
                    boundary_type, orientation, ref_point, turn_type, fly_type, gidx = (
                        key[1:]
                    )
                    if ref_point == "ctr":
                        ref_pt_desc = "body center"
                    elif ref_point == "edge":
                        ref_pt_desc = "edge of ellipse"
                    if turn_type == "all":
                        turn_type_desc = ""
                    else:
                        turn_type_desc = f", {turn_type}"
                    title_str = (
                        f"Training {training}, {fly_type} fly, sharp turns, "
                        f"{ref_pt_desc}{turn_type_desc}\nn = {num_samples}"
                    )
                    filename = (
                        f"turn_directionality_t{training}_{fly_type}_{boundary_type}_"
                        f"{orientation}_{ref_point}_{turn_type}.png"
                    )
                    plot_pie_chart(sizes, labels, title_str, filename)
