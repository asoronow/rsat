import numpy as np
import os
from pathlib import Path
import pickle
import argparse
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from main import ROI, loadROI
import csv
from mpl_toolkits.mplot3d import Axes3D


def plot_heatmap_3d(roi, output_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = np.indices(np.array(roi.shape) + 1)

    # Normalize your ROI data for the colormap
    normalized_roi = roi / np.max(roi)

    # Get the 'viridis' colormap
    viridis = plt.get_cmap("viridis")

    # Map the data values to colors
    colors = viridis(normalized_roi)

    # Plot voxels with colors mapped from the data values
    ax.voxels(x, y, z, roi, facecolors=colors)

    plt.savefig(output_path, dpi=300)
    plt.close()


def plotVerticalLine(experiments, output_path):
    """
    Creates vertical line plots based of the average intensity of each coordinate
    in an ROI across all brains in an age group.
    """

    all_animals = {}
    sum_acitivity = {}
    # Initialize meanGrids and stderrorGrids with zero arrays
    for age_group in experiments.keys():
        for animal in experiments[age_group].keys():
            for roi in experiments[age_group][animal].keys():
                # make a 3d array of all the grids
                if roi not in all_animals:
                    all_animals[roi] = []
                # convert samples to arrays
                all_animals[roi].append(np.array(experiments[age_group][animal][roi]))

    # Make the normalized grids and stderror grids
    print("Plotting vertical line plots...")

    roi_layout = [
        [None, "VISal", "VISrl", "VISa", "RSPagl"],
        ["VISli", "VISl", None, "VISam", "RSPd"],
        ["VISpor", "VISpl", None, "VISpm", "RSPv"],
    ]

    fig, axes = plt.subplots(
        3, 5, figsize=(15, 10)
    )  # Adjusted for a 3-row, 5-column layout

    max_roi_count = 0
    all_mean_data = {}
    all_std_err = {}
    for row_idx, row in enumerate(roi_layout):
        for col_idx, roi in enumerate(row):
            if roi:
                ax = axes[row_idx, col_idx]

                ax.set_title(roi)
                ax.set_xlabel("Axon coverage (A.U.)")
                ax.set_ylabel("Depth from pial surface (A.U.)")
                ax.set_ylim(0, 100)
                ax.set_xlim(0, 1)
                ax.set_yticks([0, 100])
                ax.set_yticklabels([1, 0])
                roi_key = roi.lower()

                if roi_key not in all_animals:
                    roi_data = np.zeros((101))
                    all_mean_data[roi_key] = roi_data
                    all_std_err[roi_key] = roi_data
                else:
                    roi_data = all_animals[roi_key]
                    normal_data = np.zeros((len(roi_data), 101))
                    for i, cube in enumerate(roi_data):
                        # plot the 3d cube with counts
                        # plot_heatmap_3d(cube, roi_key)

                        if sum_acitivity.get(f"Animal_{i}") is None:
                            sum_acitivity[f"Animal_{i}"] = {}
                        # Sum project the grids
                        sum_projected = np.sum(cube, axis=0)
                        # cv2.imwrite(f"{roi_key}_{i}.png", sum_projected)
                        sum_projected = np.sum(sum_projected, axis=0)
                        # Set all nans to 0
                        sum_projected = np.nan_to_num(sum_projected)
                        sum_projected = sum_projected / np.max(sum_projected)
                        if np.max(sum_projected) > max_roi_count:
                            max_roi_count = np.max(sum_projected)
                        # Get area under the curve
                        sum_acitivity[f"Animal_{i}"][roi_key] = sum_projected
                        normal_data[i] = sum_projected

                    mean_data = np.mean(normal_data, axis=0)
                    std_err = np.std(normal_data, axis=0) / np.sqrt(
                        normal_data.shape[0]
                    )

                    all_mean_data[roi_key] = mean_data
                    all_std_err[roi_key] = std_err
            else:
                fig.delaxes(axes[row_idx, col_idx])

    for row_idx, row in enumerate(roi_layout):
        for col_idx, roi in enumerate(row):
            if roi:
                ax = axes[row_idx, col_idx]
                roi_key = roi.lower()
                mean_data = all_mean_data[roi_key]
                std_err = all_std_err[roi_key]

                ax.barh(
                    np.arange(101),
                    mean_data,
                    color="red",
                )
                # Plot the standard error
                ax.fill_betweenx(
                    np.arange(101),
                    mean_data - std_err,
                    mean_data + std_err,
                    color="black",
                    alpha=0.3,
                )

    # write each animal's sum acitivity for each roi
    with open(output_path / "sum_activity.csv", "w") as f:
        writer = csv.writer(f)
        # normalize and sum the activity
        sum_acitivity = {
            animal: {roi: np.sum(data) for roi, data in roi_data.items()}
            for animal, roi_data in sum_acitivity.items()
        }
        for animal, roi_data in sum_acitivity.items():
            writer.writerow([""] + list(sum_acitivity[animal].keys()))
            writer.writerow([animal] + list(roi_data.values()))

    plt.tight_layout()
    plt.savefig(
        output_path / f"combined_{age_group}.svg",
        format="svg",
        dpi=600,
        transparent=True,
    )


def process_ROI(roi, coordinateNorms):
    roi_name = roi.name.lower()

    if roi_name not in coordinateNorms:
        coordinateNorms[roi_name] = []

    min_x = np.min([i for i, j in roi.intensity.keys()])
    min_y = np.min([j for i, j in roi.intensity.keys()])

    # Temporary grid for the current ROI slice
    grid = np.zeros((101, 101))

    for i, j in roi.intensity.keys():
        l_x = min(i - min_x, roi.mask.shape[0] - 1)
        l_y = min(j - min_y, roi.mask.shape[1] - 1)
        grid[i, j] = roi.mask[l_x, l_y]

    coordinateNorms[roi_name].append(grid)


if __name__ == "__main__":
    """
    Processes a directory of directories of ROIs. Each subdirectory is a different brain.

    E.g.:
    Age_Group1/
        Brain1/
            ROI1.pkl
            ROI2.pkl
            ...
        Brain2/
            ROI1.pkl
            ROI2.pkl
            ...
        ...
    Age_Group2/
        ...
    """
    # Setup arguments
    parser = argparse.ArgumentParser(description="Process some ROIs.")
    parser.add_argument("--input", type=str, help="Input ROI pkl file or directory")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument(
        "--regraph",
        action="store_true",
        help="Regraph the data from the raw experiments",
        default=False,
    )
    args = parser.parse_args()

    if args.input == None:
        raise Exception("No input directory provided")
    elif not os.path.exists(args.input):
        raise Exception("Input directory does not exist")
    elif not os.path.isdir(args.input):
        raise Exception("Input is not a directory")

    if args.output == None:
        raise Exception("No output directory provided")
    elif not os.path.exists(args.output):
        raise Exception("Output directory does not exist")
    elif not os.path.isdir(args.output):
        raise Exception("Output is not a directory")

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.regraph:
        # find the experiments pkl
        experiments_pkl = None
        with open(input_path / f"raw_experiments_{input_path.stem}.pkl", "rb") as f:
            experiments_pkl = pickle.load(f)
        plotVerticalLine(experiments_pkl, output_path)
        quit()

    # Get subdirectories from input
    subs_dirs = [
        input_path / sub_dir
        for sub_dir in os.listdir(input_path)
        if os.path.isdir(input_path / sub_dir)
    ]
    experiments = {input_path.stem: {}}
    num_rois = len([x for x in Path(input_path).glob("**/*.pkl") if x.is_file()])

    c = 0
    for exp_dir in subs_dirs:
        animal_name = os.path.basename(exp_dir)
        coordinateNorms = {}

        for roi in loadROI(exp_dir):
            if roi is None:
                print("No ROIs found, exiting...")
                break

            # Run preprocessing on the ROI
            try:
                print(f"Preprocessing {roi.filename} [{c + 1}/{num_rois}]")
                roi.create_axon_mask()
                roi.normalize()

                process_ROI(roi, coordinateNorms)
            except Exception as e:
                print(f"Error processing ROI: {roi.filename}. Error: {str(e)}")
                continue

            c += 1

        # Add to experiments dict under age group (args.input) and animal name
        experiments[input_path.stem][animal_name] = coordinateNorms

    if not args.regraph:
        # save raw experiments as a pickle file
        with open(
            Path(
                args.output.strip(),
                f"raw_experiments_{input_path.stem}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(experiments, f)

    plotVerticalLine(experiments, output_path)
