import os
import re
import random
from collections import defaultdict


DATA_DIR = 'data/good' 
target_count = 2500
random.seed(42)


def parse_site_video(filename):
    match = re.match(r'([a-zA-Z0-9_]+)_[a-zA-Z0-9]+_\d+_(\d+)\.png', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

groups = defaultdict(list)
all_images = [f for f in os.listdir(DATA_DIR) if f.endswith('.png')]

for fname in all_images:
    site, vid = parse_site_video(fname)
    if site and vid:
        groups[(site, vid)].append(fname)


total_images = sum(len(imgs) for imgs in groups.values())
if total_images <= target_count:
    print(f"Total images ({total_images}) <= target ({target_count}), nothing to delete.")
    exit(0)


allocations = {}
remaining = target_count
for (site, vid), imgs in groups.items():
    n = len(imgs)
    keep = int(n * target_count / total_images)
    allocations[(site, vid)] = keep
    remaining -= keep


if remaining > 0:
    # Sort groups by largest remainder
    remainders = sorted(groups.items(), key=lambda x: -(len(x[1]) * target_count / total_images - allocations[x[0]]))
    for (site, vid), imgs in remainders:
        if remaining == 0:
            break
        allocations[(site, vid)] += 1
        remaining -= 1


keep_set = set()
for (site, vid), imgs in groups.items():
    k = allocations[(site, vid)]
    if k > 0:
        keep_set.update(random.sample(imgs, min(k, len(imgs))))


for fname in all_images:
    if fname not in keep_set:
        os.remove(os.path.join(DATA_DIR, fname))
        print(f"Deleted {fname}")

print(f"Kept {len(keep_set)} images. Done.")
