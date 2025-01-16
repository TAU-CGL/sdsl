import pickle

import nml_bag

if __name__ == "__main__":
    reader = nml_bag.Reader("traj/traj_0.mcap", topics=["/scan_idx", "/pose"])
    records = []
    for record in reader:
        records.append(record)
    with open("traj_raw.pkl", "wb") as fp:
        pickle.dump(records, fp)
