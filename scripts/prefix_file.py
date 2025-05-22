import os
import argparse

dir_path = '/media/femiloye/SSD8/femiloye/datasets/visual_reid/pyramid_in_transformer/data_2.0/selected_crops/bad/pdgr'

def add_prefix_to_files(directory, prefix):
    for fname in os.listdir(directory):
        src = os.path.join(directory, fname)
        if os.path.isfile(src) and not fname.startswith(prefix + "_"):
            dst = os.path.join(directory, f"{prefix}_{fname}")
            os.rename(src, dst)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add prefix to all files in a directory.")
    parser.add_argument('--dir', required=True, help='Directory containing files')
    parser.add_argument('--prefix', required=True, help='Prefix to add')
    args = parser.parse_args()
    add_prefix_to_files(args.dir, args.prefix)

