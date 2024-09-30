import nrrd
import pickle
import numpy as np

def main():
    atlas, _ = nrrd.read("/Users/alec/.belljar/nrrd/annotation_10.nrrd")
    # read structure_graph
    with open("./structure_map.pkl", "rb") as f:
        structure_map = pickle.load(f)


    to_analyze = [
        "RSPv",
        "RSPd",
        "RSPagl",
        "VISpm",
        "VISl",
        "VISp",
        "VISrl",
        "VISal",
        "VISpor",
        "VISam",
        "VISli",
    ]

    proportions = {
        k: 0 for k in to_analyze
    }

    parent_ids = {}

    for atlas_id, data in structure_map.items():
        if data["acronym"] in to_analyze:
            # get all structures where atlas_id is in the id_path
            parent_ids[data["acronym"]] = atlas_id

    for atlas_id, data in structure_map.items():
        for parent_id in parent_ids.values():
            if parent_id in [np.uint32(val) for val in data["id_path"].split("/")]:
                proportions[structure_map[parent_id]["acronym"]] += (atlas == atlas_id).sum()

    total = np.sum(list(proportions.values()))
    for k, v in proportions.items():
        proportions[k] = v / total

    print(proportions)


if __name__ == "__main__":
    main()