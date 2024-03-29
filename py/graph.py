import numpy as np
import os
from pathlib import Path
import pickle
import argparse
import matplotlib.pyplot as plt
from main import ROI, loadROI, load_roi_from_file
import multiprocessing
from multiprocessing import Pool
import csv

def plotVerticalLine(experiments, output_path):
    """
    Creates vertical line plots based of the average intensity of each coordinate
    in an ROI across all brains in an age group.
    """

    all_animals = {}
    animal_sums = {}
    normalization_counts = None
    if args.counts:
        normalization_counts = {}
        with open(args.counts, "r") as f:
            reader = csv.reader(f)
            animals = next(reader)
            counts = next(reader)
            for i, animal in enumerate(animals):
                normalization_counts[animal] = float(counts[i])
                
    # Initialize meanGrids and stderrorGrids with zero arrays
    for age_group in experiments.keys():
        for animal in experiments[age_group].keys():
            for roi in experiments[age_group][animal].keys():
                # n dim array of all the rois from this experiement
                if animal not in animal_sums:
                    animal_sums[animal] = {}

                if roi not in all_animals:
                    all_animals[roi] = []
                all_animals[roi].append(experiments[age_group][animal][roi])

    # Make the normalized grids and stderror grids
    print("Plotting vertical line plots...")

    roi_layout = [
        [None, "VISal", "VISrl", "VISa", "RSPagl"],
        ["VISli", "VISl", None, "VISam", "RSPd"],
        ["VISpor", "VISpl", None, "VISpm", "RSPv"],
    ]
    roi_linear = [
        "RSPv",
        "RSPd",
        "RSPagl",
        "VISpm",
        "VISam",
        "VISa",
        "VISrl",
        "VISal",
        "VISl",
        "VISli",
        "VISpl",
        "VISpor",
    ]
    fig, axes = plt.subplots(
        3, 5, figsize=(15, 10)
    )

    max_roi_count = 0
    all_total_data = {}
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
                    roi_data = [[roi.mask for roi in animal] for animal in all_animals[roi_key]]
                    roi_area = [[roi.area for roi in animal] for animal in all_animals[roi_key]]
                    animal_names = [Path(animal[0].filename).stem.split("_")[0] for animal in all_animals[roi_key]]
                    normal_data = np.zeros((len(roi_data), 101))
                    for i, cube in enumerate(roi_data):
                        # plot the 3d cube with counts
                        # plot_heatmap_3d(cube, roi_key)
                        # Sum project the grids
                        sum_projected = np.sum(cube, axis=0)
                        # cv2.imwrite(f"{roi_key}_{i}.png", sum_projected)
                        sum_projected = np.sum(sum_projected, axis=0)
                        # Set all nans to 0
                        sum_projected = np.nan_to_num(sum_projected)
                        if np.max(sum_projected) > max_roi_count:
                            max_roi_count = np.max(sum_projected)
                        sum_projected = sum_projected[::-1]

                        if normalization_counts is not None:               
                            animal_sums[animal_names[i]][roi_key] = sum_projected / normalization_counts[animal_names[i]]
                            # print(f"Sum projected {roi_key} for {animal_names[i]}: {np.sum(sum_projected)}")
                            normal_data[i] = sum_projected / normalization_counts[animal_names[i]]
                        else:
                            animal_sums[animal_names[i]][roi_key] = sum_projected
                            normal_data[i] = sum_projected
                    
                    # linear graphing
                    all_total_data[roi_key] = [np.sum(data) for data in normal_data]
                    # data for scatter plots
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
                mean_data = all_mean_data[roi_key] / max_roi_count
                std_err = all_std_err[roi_key] / max_roi_count
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

    plt.tight_layout()
    plt.savefig(
        output_path / f"combined_{age_group}.svg",
        format="svg",
        dpi=600,
        transparent=True,
    )

    # box plots for each roi
    fig, ax = plt.subplots(
        1, 1, figsize=(10, 5))

    # reorder all_total_data to match the order of roi_linear
    reorder = {}
    for roi in roi_linear:
        roi = roi.lower()
        if roi in all_total_data:
            reorder[roi] = all_total_data[roi]
        else:
            reorder[roi] = [0]

    ax.plot([np.mean(data) for data in reorder.values()], marker="o", color="red")
    ax.errorbar(
        range(len(reorder)),
        [np.mean(data) for data in reorder.values()],
        yerr=[(np.std(data) / np.sqrt(len(data))) for data in reorder.values()],
        fmt="o",
        color="red",
    )
    ax.set_xticks(range(len(roi_linear)))
    ax.set_xticklabels(roi_linear)
    ax.set_ylabel("Axon coverage (A.U.)")
    ax.set_xlabel("Region")
    ax.set_title("Axon coverage in each region")

    plt.tight_layout()
    plt.savefig(
        output_path / f"boxplot_{age_group}.svg",
        format="svg",
        dpi=600,
        transparent=True,
    )

    # write each animal's sum total for each roi
    with open(output_path / "sum_activity.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Animal"] + roi_linear)
        for animal in animal_sums.keys():
            row_data = []
            for roi in roi_linear:
                if roi.lower() in animal_sums[animal]:
                    row_data.append(np.sum(animal_sums[animal][roi.lower()]))
                else:
                    row_data.append(0)
            writer.writerow([animal] + row_data)

def process_roi(roi_path):
    """
    This function is designed to be run in parallel. It loads a ROI,
    processes it by calling create_axon_mask, and then returns the processed ROI.
    It's wrapped in a try-except block to handle exceptions that might occur during processing.
    """
    try:
        # Assuming loadROI loads a single ROI from the given path.
        roi = load_roi_from_file(roi_path)
        animal_name = Path(roi_path).stem.split("_")[0]
        if roi is not None:
            print(f"Preprocessing {roi.filename}")
            roi.create_axon_mask()
            return animal_name, roi.name.lower(), roi
    except Exception as e:
        print(f"Error processing ROI: {roi_path}. Error: {str(e)}")
    return None, None, None

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
    parser.add_argument("--counts", type=str, help="Path to counts csv file for normalization")
    parser.add_argument(
        "--regraph",
        action="store_true",
        help="Regraph the data from the raw experiments",
        default=False,
    )
    args = parser.parse_args()

    # escape backslashes in input and output paths
    args.input = args.input.strip().replace("\\", "\\\\")
    args.output = args.output.strip().replace("\\", "\\\\")
    if args.counts:
        args.counts = args.counts.strip().replace("\\", "\\\\")

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
    roi_paths = [x.absolute() for x in Path(input_path).glob("**/*.pkl") if x.is_file()]
    num_rois = len(roi_paths)
    c = 0
    with Pool(multiprocessing.cpu_count()) as pool:
        results = pool.map(process_roi, roi_paths)

    # Add to experiments dict under age group (args.input) and animal name
    for animal_name, roi_name, roi in results:
        if roi is not None:
            if animal_name not in experiments[input_path.stem]:
                experiments[input_path.stem][animal_name] = {}
            if roi_name not in experiments[input_path.stem][animal_name]:
                experiments[input_path.stem][animal_name][roi_name] = []
            
            experiments[input_path.stem][animal_name][roi_name].append(roi)


    if not args.regraph:
        # save raw experiments as a pickle file
        with open(
            Path(
                args.output.strip(),
                f"raw_experiments_{input_path.stem}.pkl",
            ),
            "wb",
        ) as f:
            # drop intensity data from the experiments
            for age_group in experiments.keys():
                for animal in experiments[age_group].keys():
                    for roi in experiments[age_group][animal].keys():
                        for i in range(len(experiments[age_group][animal][roi])):
                            experiments[age_group][animal][roi][i].intensity = None
            pickle.dump(experiments, f)

    plotVerticalLine(experiments, output_path)
