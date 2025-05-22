import os
import shutil
import argparse
from collections import defaultdict

SITE_ALIAS = {
    'jochen': 'jsa',
    'praia-da-grama': 'pdgr',
    'skudin-surf': 'skd',
    'vilnius': 'vil',
    'wellenwerk': 'well'
}

def parse_video_id(filename):
    # Assumes video_id is the last underscore-separated part before .png
    return filename.rsplit('_', 1)[-1].replace('.png', '')

def main(src_dir, dst_dir, n_crops):
    for site, alias in SITE_ALIAS.items():
        site_path = os.path.join(src_dir, site)
        if not os.path.isdir(site_path):
            continue
        # Group crops by video_id
        video_groups = defaultdict(list)
        for fname in os.listdir(site_path):
            if fname.endswith('.png'):
                vid = parse_video_id(fname)
                video_groups[vid].append(fname)
        # For each video, select N crops and copy
        for vid, crops in video_groups.items():
            selected = crops[:n_crops]
            for crop in selected:
                src_crop = os.path.join(site_path, crop)
                dst_crop = os.path.join(dst_dir, f"{alias}_{crop}")
                shutil.copy2(src_crop, dst_crop)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select and copy N crops per video per site.")
    parser.add_argument("dst_dir", help="Destination directory for selected crops")
    parser.add_argument("n", type=int, help="Number of crops to select per video")
    args = parser.parse_args()
    os.makedirs(args.dst_dir, exist_ok=True)
    main(os.getcwd(), args.dst_dir, args.n)