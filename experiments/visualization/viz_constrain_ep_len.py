import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-paths", type=str, nargs="+", required=True)
    parser.add_argument("--labels", type=str, nargs="+", required=True)
    parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()

    colors = ["C{}".format(i) for i in range(len(args.labels))]
    success_rates = []

    fig, ax = plt.subplots(1, 2, figsize=(8, 2.4))
    for i, (data_path, label) in enumerate(zip(args.data_paths, args.labels)):
        ep_lens = np.loadtxt(data_path)
        success_rate = np.mean(ep_lens >= 500)
        print("{}: {:.2f}".format(label, success_rate))
        success_rates.append(success_rate)

        color = colors[i]

        ax[0].scatter(i, np.mean(ep_lens), color=color, s=4, label=label)
        ax[0].errorbar(
            i, np.mean(ep_lens), yerr=np.std(ep_lens), fmt="o", color="black"
        )
        bodies = ax[0].violinplot(
            ep_lens, positions=[i], showmeans=False, showextrema=False
        )
        for body in bodies["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.5)
        ax[0].set_ylabel("Episode length")
        ax[0].tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )  # labels along the bottom edge are off

    # pos = np.arange(len(args.labels))
    # ax.set_xticks(pos)
    # ax.set_xticklabels(args.labels)

    # patches = [mpatches.Patch(color='C{}'.format(i), label=label) for i, label in enumerate(args.labels)]
    # legend = ax[0].legend(
    #     loc='upper center', bbox_to_anchor=(0.5, -0.12),
    #     fancybox=True, shadow=True, ncol=3
    # )

    ax[1].set_ylim(0.0, 1.25)
    ax[1].bar(np.arange(len(args.labels)), success_rates, color=colors)
    ax[1].axhline(1.0, color="black", linestyle="--")
    ax[1].set_ylabel("Safety success rate")
    ax[1].tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(
        lines, labels, loc="upper center", ncol=3, fancybox=True, shadow=True
    )
    for lh in legend.legendHandles:
        lh._sizes = [30]

    fig.tight_layout()
    fig.show()
    input("Press Enter to continue...")
    fig.savefig(args.save_path + "_bar.png")
