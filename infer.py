import argparse
import os
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--input_dir', type=str, required=True)
args = parser.parse_args()

os.makedirs('predictions/good', exist_ok=True)
os.makedirs('predictions/bad', exist_ok=True)

model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
model.eval()
model = model.cuda() if torch.cuda.is_available() else model

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
for fname in tqdm(files, desc='Inferencing'):
    fpath = os.path.join(args.input_dir, fname)
    img = Image.open(fpath).convert('RGB')
    x = transform(img).unsqueeze(0)
    x = x.cuda() if torch.cuda.is_available() else x
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)
        conf, pred = prob.max(1)
        label = pred.item()
        conf_score = conf.item()
    # Draw confidence score
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    draw.text((5, 5), f"{conf_score:.2f}", fill=(255,0,0), font=font)
    out_dir = 'predictions/good' if label == 1 else 'predictions/bad'
    img.save(os.path.join(out_dir, fname))
