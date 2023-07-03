import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-paths", type=str, nargs="+", required=True)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    args = parser.parse_args()

    # Plot trajectory
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))

    if args.labels is not None:
        assert len(args.labels) == len(args.data_paths)

    for path in args.data_paths:
        data = np.loadtxt(path)
        ts = np.arange(data.shape[0])
        path = pathlib.Path(path)
        if args.labels is not None:
            label = args.labels.pop(0)
        else:
            label = path.stem
        ax.plot(ts, data[:, 0], label=label)

    ax.axhline(y=2.4, color="black", linestyle="-")
    ax.axhline(y=-2.4, color="black", linestyle="-")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=True,
        ncol=3,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("X-Position")
    fig.tight_layout()

    if args.save_path is not None:
        fig.savefig(args.save_path)
    else:
        fig.show()
        input("Press Enter to continue...")
