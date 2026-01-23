ROOT_DIR = "./results/neural_network_vlm/thinning_kt"

import os

for name in os.listdir(ROOT_DIR):
    old_path = os.path.join(ROOT_DIR, name)

    # only deal with directories
    if not os.path.isdir(old_path):
        continue

    # rename step_size_200 -> step_size_150
    if "step_num_200" in name:
        new_name = name.replace("step_num_200", "step_num_150")
        new_path = os.path.join(ROOT_DIR, new_name)

        if os.path.exists(new_path):
            print(f"Skip (target exists): {new_name}")
            continue

        os.rename(old_path, new_path)
        print(f"Renamed:\n  {name}\n→ {new_name}\n")
