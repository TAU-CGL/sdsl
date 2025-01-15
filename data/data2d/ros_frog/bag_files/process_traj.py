import nml_bag

if __name__ == "__main__":
    reader = nml_bag.Reader("traj/traj_0.mcap", topics=["/scan_idx", "/pose"])
    for record in reader:
        print(record)