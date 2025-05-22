import argparse
import os
import shutil
import random
import re
from collections import defaultdict

def get_site(filename):
    match = re.match(r'([a-zA-Z0-9_]+)_', filename)
    if match:
        return match.group(1)
    return None

def stratified_site_split(src_dir, dst_dir, test_ratio):
    class_counts = {cls: {'train': 0, 'test': 0} for cls in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, cls))}
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    for cls in classes:
        class_dir = os.path.join(src_dir, cls)
        files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        site_to_files = defaultdict(list)
        for f in files:
            site = get_site(f)
            if site:
                site_to_files[site].append(f)
        # Flatten all files, but keep site info
        all_files = []
        for site, flist in site_to_files.items():
            for f in flist:
                all_files.append((site, f))
        random.shuffle(all_files)
        n_total = len(all_files)
        n_test = int(n_total * test_ratio)
        test_set = set()
        site_counts = defaultdict(int)
        # Stratified: try to keep site proportions in test set
        for site, f in all_files:
            if len(test_set) < n_test:
                test_set.add(f)
                site_counts[site] += 1
            else:
                break
        for site, flist in site_to_files.items():
            for f in flist:
                split_name = 'test' if f in test_set else 'train'
                split_cls_dir = os.path.join(dst_dir, split_name, cls)
                os.makedirs(split_cls_dir, exist_ok=True)
                shutil.copy2(os.path.join(class_dir, f), os.path.join(split_cls_dir, f))
                class_counts[cls][split_name] += 1
    for cls in class_counts:
        print(f"train/{cls}: {class_counts[cls]['train']} images")
        print(f"test/{cls}: {class_counts[cls]['test']} images")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stratified train/test split by site for ImageFolder datasets.')
    parser.add_argument('split_ratio', type=float, help='Proportion of data to use for test (e.g., 0.32)')
    parser.add_argument('output_dir', type=str, help='Directory to save the split dataset')
    args = parser.parse_args()
    random.seed(42)
    stratified_site_split('data', args.output_dir, args.split_ratio)
    print('Done.')
