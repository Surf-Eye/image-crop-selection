from PIL import Image
import os

def check_images(root):
    for split in ['train', 'test']:
        for cls in os.listdir(os.path.join(root, split)):
            cls_dir = os.path.join(root, split, cls)
            for fname in os.listdir(cls_dir):
                fpath = os.path.join(cls_dir, fname)
                try:
                    with Image.open(fpath) as img:
                        img.verify()
                except Exception:
                    print(f"Removing corrupted file: {fpath}")
                    os.remove(fpath)

if __name__ == '__main__':
    check_images('data')