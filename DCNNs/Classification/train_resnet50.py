# coding=UTF-8  

import argparse
import os
import numpy as np
import time
import torch
import torchvision
import torchvision.models as models  
import torchvision.datasets as datasets
from torch import nn, optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR


#parameter settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser("""Image classification!""")
parser.add_argument('--train_path', type=str, default='trainingset',
                    help="""image dir path default: 'ImageNet/origin_data/trainingset'.""")
parser.add_argument('--test_path', type=str, default='testingset',
                    help="""image dir path default: 'ImageNet/origin_data/testingset'.""")
parser.add_argument('--epochs', type=int, default=10,
                    help="""Epoch default:50.""")
parser.add_argument('--batch_size', type=int, default=64,
                    help="""Batch_size default:64.""")
parser.add_argument('--lr', type=float, default=0.1,
                    help="""learing_rate. Default=0.1""")
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                         help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--num_classes', type=int, default=16,
                    help="""num classes""")
parser.add_argument('--model_path', type=str, default='model/pytorch/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='resnet50_origin.pth',
                    help="""Model name.""")
parser.add_argument('--display_epoch', type=int, default=1)

args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

transform = transforms.Compose([

    transforms.ToTensor(),  
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])

# Load data

train_datasets = datasets.ImageFolder(root=args.train_path, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
                                           batch_size=args.batch_size,
                                           shuffle=True)

test_datasets = datasets.ImageFolder(root=args.test_path, transform=transform)

test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                          batch_size=args.batch_size,
                                          shuffle=True)


def train():
    print(f"Train numbers:{len(train_datasets)}")

    model = models.resnet50().to(device)

    model.fc = nn.Linear(2048, args.num_classes).to(device)
    #print(model)
    print(f"model_name:{args.model_name}.")
    print(f"Learning Rate:{args.lr}.")
    print(f"Momentum:{args.momentum}.")
    print(f"train_path:{args.train_path}.")
    # cast
    cast = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.0027,0.0112,0.0203,0.0133,0.0047,0.0099,0.0248,0.0164,0.0083,0.0042,0.0326,0.2694,0.0110,0.0157,0.0307,0.5246])).float()).to(device)
    # Optimization
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        scheduler.step(epoch)
        model.train()
        # start time
        start = time.time()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = cast(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % args.display_epoch == 0:
            end = time.time()
            print(f"Epoch [{epoch}/{args.epochs}], "
                  f"Loss: {loss.item():.8f}, "
                  f"Time: {(end-start) * args.display_epoch:.1f}sec!")

            model.eval()

            correct_prediction = 0.
            total = 0
            for images, labels in test_loader:
                # to GPU
                images = images.to(device)
                labels = labels.to(device)
                # print prediction
                outputs = model(images)
                # equal prediction and acc
                _, predicted = torch.max(outputs.data, 1)
                # val_loader total
                total += labels.size(0)
                # add correct
                correct_prediction += (predicted == labels).sum().item()

            print(f"Acc: {(correct_prediction / total):4f}")

    # Save the model checkpoint
    torch.save(model, args.model_path + args.model_name)
    print(f"Model save to {args.model_path + args.model_name}.")


if __name__ == '__main__':
    train()
	