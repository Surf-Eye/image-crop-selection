import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import wandb
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data', help='Root data dir')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--finetune', action='store_true', help='Full finetune if set, else feature extraction')
parser.add_argument('--wandb_project', type=str, default='crops_classification')
args = parser.parse_args()

run_name = f"resnet50_lr{args.lr}_{args.optimizer}_{'finetune' if args.finetune else 'feature'}"
run_dir = os.path.join('checkpoints', run_name)
os.makedirs(run_dir, exist_ok=True)
wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(f"{args.data_dir}/train", transform=train_tfm)
test_ds = datasets.ImageFolder(f"{args.data_dir}/test", transform=test_tfm)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

if not args.finetune:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

model = model.cuda() if torch.cuda.is_available() else model

criterion = nn.CrossEntropyLoss()
if args.optimizer == 'adam':
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
else:
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9)

best_acc = 0.0
best_path = os.path.join(run_dir, 'best_model.pth')

for epoch in range(args.epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda() if torch.cuda.is_available() else (x, y)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    train_loss = running_loss / total
    train_acc = correct / total
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda() if torch.cuda.is_available() else (x, y)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item() * x.size(0)
            _, pred = out.max(1)
            val_correct += pred.eq(y).sum().item()
            val_total += y.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    wandb.log({'epoch': epoch+1, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'precision': prec, 'recall': rec})
    print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_path)
        # Save confusion matrix plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_ds.classes)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(run_dir, 'confusion_matrix.png'))
        plt.close()
        # Save precision and recall as bar plot
        plt.bar(['precision', 'recall'], [prec, rec])
        plt.ylim(0, 1)
        plt.title('Precision and Recall')
        plt.savefig(os.path.join(run_dir, 'precision_recall.png'))
        plt.close()

wandb.finish()
