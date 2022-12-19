import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import sys
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import smdebug.pytorch as smd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device, hook):
    model.eval() 
    hook.set_mode(smd.modes.EVAL) # assign the debugger hook
    losses = []
    corrects=0  
    
    for data, label in test_loader:
        data = data.to(device)
        label = label.to(device)
        output=model(data)
        loss=criterion(output, label)
        _, preds = torch.max(output, 1)
        losses.append(loss.item())             # calculate running loss
        corrects += torch.sum(preds == label.data)     # calculate running corrects

    avg_loss = np.mean(losses)
    avg_acc = np.true_divide(corrects.double().cpu(), len(test_loader.dataset))
    
    logger.info('Testing Loss: {:.4f}, Accuracy: {:.4f}'.format(avg_loss, avg_acc)) # print the avg loss and accuracy values

def train(model, train_loader, valid_loader, criterion, optimizer, device, epochs, hook):

    best_loss = np.Inf #initialize best loss to infinity
    hook.set_mode(smd.modes.TRAIN) # set debugging hook
    dataset_dict={'train':train_loader, 'valid':valid_loader}
    
    for epoch in range(1, epochs+1):
        logger.info(f"Epoch:{epoch}---")
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                hook.set_mode(smd.modes.TRAIN) # set debugging hook
            else:
                model.eval()
                hook.set_mode(smd.modes.EVAL)
                
            losses = []
            corrects = 0
            
            for data, label in dataset_dict[phase]:
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                loss = criterion(output, label)
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                _, preds = torch.max(output, 1)
                
                losses.append(loss.item())
                corrects += torch.sum(preds == label.data)

            avg_loss = np.mean(losses)
            avg_acc = np.true_divide(corrects.cpu(), len(dataset_dict[phase].dataset))
            
            if phase=='valid':
                if avg_loss < best_loss:
                    best_loss = avg_loss
            
            logger.info('{} Loss: {:.4f}, Accuracy: {:.4f}, Best Loss: {:.4f}'.format(phase.capitalize(),
                                                                                 avg_loss,
                                                                                 avg_acc,
                                                                                 best_loss))
    return model
    
def net(num_classes=5):
    # load the pre-trained model - resnet50
    model = models.resnet50(pretrained=True)
    # freeze part of  model except the last linear layer 
    for param in model.parameters():
        param.requires_grad = False  
    # input size for last linear layer
    num_features = model.fc.in_features
    # update the last linear layer
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def create_data_loaders(data, batch_size):
    
    train_data_path = os.path.join(data, 'train') # dogImages/train
    test_data_path = os.path.join(data, 'test') # dogImages/test
    valid_data_path = os.path.join(data, 'validation') # dogImages/valid
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,0,0), std=(1,1,1))
    ])

    train_loader = DataLoader(
        ImageFolder(train_data_path, transform=transform),
        batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        ImageFolder(test_data_path, transform=transform),
        batch_size=batch_size)
    valid_loader = DataLoader(
        ImageFolder(valid_data_path, transform=transform),
        batch_size=batch_size)
    
    return train_loader, test_loader, valid_loader

def save_model(model, model_dir):
    logger.info("Saving the model...")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

def main(args):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    logger.info(f"Running on Device {device}")
    logger.info(f'Hyperparameters are LR: {args.lr}, Batch Size: {args.batch_size}')
    logger.info(f'Data Path: {args.data_dir}')
    
    model=net(args.num_classes)
    model = model.to(device)
    # hook
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    
    loss_criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = optim.Adagrad(model.parameters(), lr=args.lr) #using adagrad which has adaptive learning rate
    optimizer = optim.Adam(model.parameters(), lr=args.lr) #using adagrad which has adaptive learning rate
    hook.register_loss(loss_criterion)
    
    train_loader, test_loader, valid_loader = create_data_loaders(args.data_dir, args.batch_size)
    
    logger.info("Training model...")
    model=train(model, train_loader, valid_loader, loss_criterion, optimizer, device, args.epochs, hook)
    
    logger.info("Testing the model...")
    test(model, test_loader, loss_criterion, device, hook)
    
    # Saving the model
    save_model(model, args.model_dir)
    
if __name__=='__main__':
    parser=argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs (default: 5)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        metavar="N",
        help="number of classes (default: 5)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )

    # Container environment
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)