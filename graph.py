import numpy as np
import os, pathlib
import pickle
import argparse
import matplotlib.pyplot as plt
from main import ROI, loadROI
import concurrent.futures

NUM_WORKERS = 4

def plotVerticalLine(experiments):
    """
    Creates vertical line plots based of the average intensity of each coordinate
    in an ROI across all brains in an age group.
    """

    # save raw experiments as a pickle file
    with open(pathlib.Path(args.output.strip(), "raw_experiments.pkl"), "wb") as f:
        pickle.dump(experiments, f)
    
    meanGrids = {}
    stderrorGrids = {}

    # Initialize meanGrids and stderrorGrids with zero arrays
    for brain in experiments.keys():
        for roi in experiments[brain].keys():
            if roi not in meanGrids:
                meanGrids[roi] = np.zeros((101, 101))
                stderrorGrids[roi] = np.zeros((101, 101))

    # Populate meanGrids and stderrorGrids
    for brain, rois in experiments.items():
        for roi, coords in rois.items():
            for (i, j), values in coords.items():
                meanGrids[roi][i, j] = np.mean(values)
                stderrorGrids[roi][i, j] = np.std(values) / np.sqrt(len(values))


    print("Plotting vertical line plots...")

    roi_layout = [
        ["TEa", "VISal", "VISrl", "VISa", "RSPagl"],
        ["VISli", "VISl", None, "VISam", "RSPd"],
        ["VISpor", "VISpl", None, "VISpm", "RSPv"],
    ]

    fig, axes = plt.subplots(
        3, 5, figsize=(15, 10)
    )  # Adjusted for a 3-row, 5-column layout

    for row_idx, row in enumerate(roi_layout):
        for col_idx, roi in enumerate(row):
            if roi:
                roi_key = roi.lower()
                if roi_key not in list(meanGrids.keys()):
                    meanGrids[roi_key] = np.zeros((101, 101))

                ax = axes[row_idx, col_idx]

                ax.set_title(roi)
                ax.set_xlabel("Axon coverage (%)")
                ax.set_ylabel("Depth from pial surface (A.U.)")
                ax.set_ylim(0, 100)
                ax.set_xlim(0, 1)

                ax.set_yticks([0, 100])
                ax.set_yticklabels([1, 0])

                fraction_mean_rows = np.mean(meanGrids[roi_key], axis=1)
                fraction_stderror = np.mean(stderrorGrids[roi_key], axis=1)

                ax.barh(
                    np.arange(0, 101, 1),
                    fraction_mean_rows,
                    color="red",
                    alpha=0.5,
                    xerr=fraction_stderror  # adding error bars
                )


            else:
                fig.delaxes(axes[row_idx, col_idx])

    plt.tight_layout()

    plt.savefig(
        pathlib.Path(args.output.strip(), "combined_roi_plot.svg"),
        format="svg",
        dpi=600,
        transparent=True,
    )

    

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
    parser.add_argument("--regraph", action="store_true", help="Regraph the data from the raw experiments")
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

    if args.regraph:
        # find the experiments pkl
        experiments_pkl = None
        with open("raw_experiments.pkl", "rb") as f:
            experiments_pkl = pickle.load(f)

        print(experiments_pkl)

        quit()

    # Get subdirectories from input
    subs_dirs = [
        os.path.join(args.input, sub_dir)
        for sub_dir in os.listdir(args.input)
        if os.path.isdir(os.path.join(args.input, sub_dir))
    ]
    # Create a dictory of experiments which is just the subdirectories
    experiments = {os.path.basename(sub_dir): sub_dir for sub_dir in subs_dirs}
    num_rois = sum(
    [
        len([file for file in os.listdir(sub_dir) if file.endswith(".pkl")])
        for sub_dir in subs_dirs
    ]
    )
    c = 0
    for exp_dir in subs_dirs:
        coordinateNorms = {}
        outputGrids = {}
        errorGrids = {}
        for roi in loadROI(exp_dir):
            if roi is None:
                print("No ROIs found, exiting...")
                break

            # run preprocessing on the ROI
            try:
                print(f"Preprocessing {roi.filename} [{c + 1}/{num_rois}]")
                roi.normalize()
                roi.create_axon_mask()
            except:
                print("Error processing ROI: {}".format(roi.filename))
                continue

            roi_name = roi.name.lower()

            if not roi_name in list(coordinateNorms.keys()):
                coordinateNorms[roi_name] = {}

            min_x = np.min([i for i, j in roi.intensity.keys()])
            min_y = np.min([j for i, j in roi.intensity.keys()])
            # parse over the normalized coordinates adding the intensity valeus to the dictionary
            for i, j in roi.intensity.keys():
                # rescale the coordinates to the mask

                if (i, j) not in coordinateNorms[roi_name]:
                    coordinateNorms[roi_name][(i, j)] = []

                l_x = min(i - min_x, roi.mask.shape[0] - 1)
                l_y = min(j - min_y, roi.mask.shape[1] - 1)
                coordinateNorms[roi_name][(i, j)].append(roi.mask[l_x, l_y])
            c += 1

        experiments[os.path.basename(exp_dir)] = coordinateNorms
    
    plotVerticalLine(experiments)
