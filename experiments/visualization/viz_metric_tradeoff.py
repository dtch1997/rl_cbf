""" Plot the tradeoff between two metrics """

import argparse
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

METRIC1_NAME = "eval/barrier/validity_alpha_0.9"
METRIC2_NAME = "eval/barrier/coverage"

exp_name_to_label = {
    "bump_supervised_base_2M": "NOEXP",
    "bump_supervised_2M": "SIGMOID_SUP",
    "bump_2M": "SIGMOID",
    "baseline_supervised_2M": "MLP_SUP",
    "baseline_2M": "MLP",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-filepath", type=str, required=True)
    parser.add_argument("-o", "--output-filepath", type=str, default=None)
    parser.add_argument("-m1", "--metric1-name", type=str, default=METRIC1_NAME)
    parser.add_argument("-m2", "--metric2-name", type=str, default=METRIC2_NAME)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    df = pd.read_csv(args.input_filepath)
    df = df[["exp_name", "seed", "global_step", args.metric1_name, args.metric2_name]]

    # Plot
    exp_names = df["exp_name"].unique()
    fig, ax = plt.subplots()

    for exp_name in exp_names:
        sub_df = df.where(df["exp_name"] == exp_name)
        m1 = sub_df[f"{args.metric1_name}"]
        m2 = sub_df[f"{args.metric2_name}"]
        ax.scatter(m1, m2, label=exp_name_to_label[exp_name], alpha=0.1)

    ax.set_xlabel(args.metric1_name)
    ax.set_ylabel(args.metric2_name)
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        fancybox=True,
        shadow=True,
        ncol=3,
    )
    for lh in legend.legendHandles:
        lh.set_alpha(1)
    fig.tight_layout()

    if args.output_filepath is not None:
        fig.savefig(args.output_filepath)
    else:
        fig.show()
        input("Press Enter to continue...")
