ROOT_DIR = "./results/neural_network_vlm/thinning_kt"

import os

for name in os.listdir(ROOT_DIR):
    old_path = os.path.join(ROOT_DIR, name)

    # only deal with directories
    if not os.path.isdir(old_path):
        continue

    # only rename if skip_swap_False is in the folder name
    if "skip_swap_False" in name:
        new_name = name.replace("skip_swap_False", "skip_swap_True")
        new_path = os.path.join(ROOT_DIR, new_name)

        if os.path.exists(new_path):
            print(f"Skip (target exists): {new_name}")
            continue

        os.rename(old_path, new_path)
        print(f"Renamed:\n  {name}\n→ {new_name}\n")
