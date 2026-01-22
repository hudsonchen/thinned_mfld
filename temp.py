import os
import shutil

ROOT_DIR = "./results/neural_network_mmd_flow/thinning_kt"  # change this to your target directory

for name in os.listdir(ROOT_DIR):
    path = os.path.join(ROOT_DIR, name)

    # only deal with directories
    if not os.path.isdir(path):
        continue

    # skip if already has skip_swap info
    if "skip_swap_True" in name or "skip_swap_False" in name:
        continue

    # case 1: folder has __complete → rename
    if "__complete" in name:
        new_name = name.replace("__complete", "__skip_swap_False__complete")
        new_path = os.path.join(ROOT_DIR, new_name)

        if not os.path.exists(new_path):
            os.rename(path, new_path)
            print(f"Renamed:\n  {name}\n→ {new_name}\n")
        else:
            print(f"Skip rename (target exists): {new_name}")

    # case 2: no __complete → delete
    else:
        shutil.rmtree(path)
        print(f"Deleted: {name}")
