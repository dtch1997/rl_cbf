""" Plot the mean and stdev of metrics over the training history """

import argparse
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

METRIC_NAME = "eval/rollout/episode_return"

exp_name_to_label = {
    "bump_supervised_base_2M": "NOEXP",
    "bump_supervised_2M": "SIGMOID_SUP",
    "bump_2M": "SIGMOID",
    "baseline_supervised_2M": "MLP_SUP",
    "baseline_2M": "MLP",
}

metric_name_to_paper_name = {
    "eval/rollout/episode_return": "Episode return",
    "eval/rollout/episode_length": "Episode length",
    "eval/barrier/validity_alpha_0.9": "Validity",
    "eval/barrier/coverage": "Coverage",
    "eval/grid/mean_td_error": "TD error",
    "eval/constrain/mean_episode_length": "Constrained episode length",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-filepath", type=str, required=True)
    parser.add_argument("-o", "--output-filepath", type=str, default=None)
    parser.add_argument("-m", "--metric-name", type=str, default=METRIC_NAME)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    df = pd.read_csv(args.input_filepath)
    df = df[["exp_name", "seed", "global_step", args.metric_name]]
    # Calculate mean and stdev across seeds
    mean = (
        df[["exp_name", "global_step", args.metric_name]]
        .groupby(["exp_name", "global_step"])
        .mean()
    )
    stdev = (
        df[["exp_name", "global_step", args.metric_name]]
        .groupby(["exp_name", "global_step"])
        .std()
    )
    df = df.join(mean, on=["exp_name", "global_step"], rsuffix="_mean")
    df = df.join(stdev, on=["exp_name", "global_step"], rsuffix="_stdev")

    # Plot
    exp_names = df["exp_name"].unique()
    df = df[
        [
            "exp_name",
            "global_step",
            args.metric_name,
            f"{args.metric_name}_mean",
            f"{args.metric_name}_stdev",
        ]
    ]
    fig, ax = plt.subplots(figsize=(10, 5))

    for exp_name in exp_names:
        label = exp_name_to_label[exp_name]
        print(exp_name, label)
        sub_df = df.where(df["exp_name"] == exp_name)
        global_step = sub_df["global_step"]
        mean = sub_df[f"{args.metric_name}_mean"]
        stdev = sub_df[f"{args.metric_name}_stdev"]
        ax.plot(global_step, mean, label=label)
        ax.fill_between(global_step, mean - stdev, mean + stdev, alpha=0.1)
    ax.set_xlabel("Env steps")
    ax.set_ylabel(metric_name_to_paper_name[args.metric_name])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        fancybox=True,
        shadow=True,
        ncol=3,
    )
    fig.tight_layout()

    if args.output_filepath is not None:
        fig.savefig(args.output_filepath)
    else:
        fig.show()
        input("Press Enter to continue...")
