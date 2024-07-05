import numpy as np
import os
from pathlib import Path
import pickle
import argparse
import matplotlib.pyplot as plt
from main import ROI, loadROI, load_roi_from_file, correct_edges
from multiprocessing import Pool
import csv
import numpy as np
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian, median
from skimage.morphology import disk
import cv2
from scipy.ndimage import convolve

import tkinter as tk
from tkinter import Scale, Button, Label
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

TUNED_PARAMETERS = {"sigma": 0.5, "contrast": 1.0, "brightness": 0}

def plot_with_params(intensity, sigma, contrast, brightness):
    verts = list(intensity.keys())
    min_x = min(vert[1] for vert in verts)
    max_x = max(vert[1] for vert in verts)
    min_y = min(vert[0] for vert in verts)
    max_y = max(vert[0] for vert in verts)

    image = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
    mask = np.zeros_like(image)
    for vert in verts:
        y, x = vert[0] - min_y, vert[1] - min_x
        image[y, x] = intensity[vert]
        mask[y, x] = 1

    # Apply contrast and brightness adjustments
    image = np.clip(contrast * image + brightness, 0, 255).astype(np.uint8)
    image = (image / np.max(image) * 255).astype(np.uint8)
    # gauss = gaussian(image, sigma=sigma)
    gauss = gaussian(image, sigma=sigma)
    horizontal = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # s2
    vertical = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])  # s1
    edges = np.abs(convolve(gauss, horizontal, mode="reflect")) + np.abs(
        convolve(gauss, vertical, mode="reflect")
    )
    thresh = threshold_otsu(edges)
    binary = edges > thresh
    outside_points = np.argwhere(mask == 0)
    binary = correct_edges(outside_points, binary, max_distance=20)
    colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colored_image[binary == 1] = [0, 0, 255]

    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    ax = axes.ravel()
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title("Original Adjusted")
    ax[0].axis("off")

    ax[1].imshow(edges, cmap=plt.cm.gray)
    ax[1].set_title("Edge Detection")
    ax[1].axis("off")

    ax[2].imshow(binary * 255, cmap=plt.cm.gray)
    ax[2].set_title("Thresholded")
    ax[2].axis("off")

    plt.tight_layout()
    return fig

def visualize_and_tweak_roi(roi):
    def update_plot():
        sigma = sigma_scale.get()
        contrast = contrast_scale.get() / 10
        brightness = brightness_scale.get()

        fig = plot_with_params(roi.intensity, sigma, contrast, brightness)
        canvas.figure = fig
        canvas.draw()

        TUNED_PARAMETERS["sigma"] = sigma
        TUNED_PARAMETERS["contrast"] = contrast
        TUNED_PARAMETERS["brightness"] = brightness

    def close_window():
        root.quit()
        root.destroy()
        print(f"Tuned Parameters: Sigma={TUNED_PARAMETERS['sigma']}, Contrast={TUNED_PARAMETERS['contrast']}, Brightness={TUNED_PARAMETERS['brightness']}")

    root = tk.Tk()
    root.title("Axon Mask Parameter Tuning")

    sigma_scale = Scale(root, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, label="Sigma")
    sigma_scale.set(TUNED_PARAMETERS["sigma"])
    sigma_scale.pack()

    contrast_scale = Scale(root, from_=5, to=20, resolution=0.1, orient=tk.HORIZONTAL, label="Contrast")
    contrast_scale.set(TUNED_PARAMETERS["contrast"] * 10)
    contrast_scale.pack()

    brightness_scale = Scale(root, from_=-100, to=100, orient=tk.HORIZONTAL, label="Brightness")
    brightness_scale.set(TUNED_PARAMETERS["brightness"])
    brightness_scale.pack()

    update_button = Button(root, text="Update Plot", command=update_plot)
    update_button.pack()

    close_button = Button(root, text="Close", command=close_window)
    close_button.pack()

    fig = plot_with_params(roi.intensity, TUNED_PARAMETERS["sigma"], TUNED_PARAMETERS["contrast"], TUNED_PARAMETERS["brightness"])
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    root.mainloop()

def plotVerticalLine(experiments, output_path):
    """
    Creates vertical line plots based of the average intensity of each coordinate
    in an ROI across all brains in an age group.
    """

    all_animals = {}
    animal_sums = {}
    animal_areas = {}
    normalization_counts = None
    h2b_counts = None
    # Load the normalization counts
    if args.counts:
        normalization_counts = {}
        with open(args.counts, "r") as f:
            reader = csv.reader(f)
            animals = next(reader)
            counts = next(reader)
            for i, animal in enumerate(animals):
                normalization_counts[animal] = float(counts[i])

    if args.h2b:
        h2b_counts = {}
        with open(args.h2b, "r") as f:
            reader = csv.reader(f)
            # skip the header
            next(reader)
            for row in reader:
                h2b_counts[row[0]] = list(row[1:])

                
    # Initialize meanGrids and stderrorGrids with zero arrays
    for age_group in experiments.keys():
        for animal in experiments[age_group].keys():
            for roi in experiments[age_group][animal].keys():
                # n dim array of all the rois from this experiement
                if animal not in animal_sums:
                    animal_sums[animal] = {}
                    animal_areas[animal] = {}

                if roi not in all_animals:
                    all_animals[roi] = []
                all_animals[roi].append(experiments[age_group][animal][roi])

    # Make the normalized grids and stderror grids
    print("Plotting vertical line plots...")

    roi_layout = [
        [None, "VISal", "VISrl", "VISa", "RSPagl"],
        ["VISli", "VISl", None, "VISam", "RSPd"],
        ["VISpor", None, None, "VISpm", "RSPv"],
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
                            animal_areas[animal_names[i]][roi_key] = np.sum(roi_area[i])
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
            reorder[roi] = []

    ax.plot([np.mean(data) for data in reorder.values()], marker="o", color="red")
    ax.errorbar(
        range(len(reorder)),
        [np.mean(data) for data in reorder.values()],
        yerr=[(np.std(data) / np.sqrt(len(data))) for data in reorder.values()],
        fmt="o",
        color="red",
    )

    # swarm plot with all the data, different color for each animal in the same roi
    # sns.swarmplot(data=[data for data in reorder.values()], ax=ax, color="black", edgecolor="black", size=5)
    # sns.boxplot(data=[data for data in reorder.values()], ax=ax, fill=False, linecolor="black", showfliers=False, whiskerprops=dict(linestyle="--"))

    if h2b_counts:
        # h2b data is each animal's h2b counts
        # need to make means
        h2b_means = {roi.lower(): [] for roi in roi_linear}
        for animal in h2b_counts.keys():
            for roi in roi_linear:
                h2b_means[roi.lower()].append(float(h2b_counts[animal][roi_linear.index(roi)]))
        # make another y axis ticks for h2b counts
        ax2 = ax.twinx()
        ax2.set_ylabel("H2B counts")
        ax2.plot([np.mean(data) for data in h2b_means.values()], marker="o", color="green")
        ax2.errorbar(
            range(len(h2b_means)),
            [np.mean(data) for data in h2b_means.values()],
            yerr=[(np.std(data) / np.sqrt(len(data))) for data in h2b_means.values()],
            fmt="o",
            color="green",
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
    
    # write each animal's area for each roi
    with open(output_path / "area.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Animal"] + roi_linear)
        for animal in animal_areas.keys():
            row_data = []
            for roi in roi_linear:
                if roi.lower() in animal_areas[animal]:
                    row_data.append(animal_areas[animal][roi.lower()])
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
            roi.create_axon_mask(TUNED_PARAMETERS=TUNED_PARAMETERS)
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
    parser.add_argument("--tune", type=str, help="Path to a pkl to tune parameters for normalization")
    parser.add_argument(
        "--regraph",
        action="store_true",
        help="Regraph the data from the raw experiments",
        default=False,
    )
    parser.add_argument(
        "--h2b",
        type=str,
        help="Path to h2b counts csv file for normalization",
    )
    args = parser.parse_args()

    # escape backslashes in input and output paths
    args.input = args.input.strip().replace("\\", "\\\\")
    args.output = args.output.strip().replace("\\", "\\\\")
    if args.counts:
        args.counts = args.counts.strip().replace("\\", "\\\\")
    if args.h2b:
        args.h2b = args.h2b.strip().replace("\\", "\\\\")

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

    if args.tune:
        # load individual pkl
        to_tune = load_roi_from_file(args.tune)


        visualize_and_tweak_roi(to_tune)

        while True:
            should_continue = input("Do you want to continue? (y/n): ")
            if should_continue.lower() == "y":
                break
            elif should_continue.lower() == "n":
                quit()
            else:
                print("Please enter either 'y' or 'n'.")

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
    with Pool(8) as pool:
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
