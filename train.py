import argparse
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets,transforms,models
import os
from time import time
from functions import data_loaders,train_model
from torch.optim import lr_scheduler

def main():
    start_time=time()
    in_arg = get_input_args()
    #check_command_line_arguments(in_arg)
    images_datasets=data_loaders(in_arg.data_dir)
    if in_arg.gpu is True:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            print("Using CPU since GPU is not available")
    else:
        device = torch.device('cpu')
    
    model = getattr(models, in_arg.arch)(pretrained=True)
    for param in model.features.parameters():
        param.require_grad = False
    output_classes=102
    hidden_units=2000
    if in_arg.arch=='vgg16':
        classifier=nn.Sequential(OrderedDict([
            ('fc1',nn.Linear(25088,4096)),
            ('relu',nn.ReLU()),
            ('dropout',nn.Dropout(p=0.5)),
            ('fc2',nn.Linear(4096,in_arg.hidden_units)),
            ('relu',nn.ReLU()),
            ('dropout',nn.Dropout(p=0.5)),
            ('fc3',nn.Linear(in_arg.hidden_units,output_classes))
            ]))
        model.classifier=classifier
    if in_arg.arch=='alexnet':
        classifier=nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(p=0.5)),
            ('fc1',nn.Linear(9216,4096)),
            ('relu',nn.ReLU()),
            ('dropout',nn.Dropout(p=0.5)),
            ('fc2',nn.Linear(4096,in_arg.hidden_units)),
            ('relu',nn.ReLU()),
            ('fc3',nn.Linear(in_arg.hidden_units,output_classes))
            ]))
        model.classifier=classifier
    model=model.to(device)
    criterion = nn.CrossEntropyLoss()
    l_r=in_arg.learning_rate
    epochs=in_arg.epochs
    optimizer = optim.SGD(model.classifier.parameters(), lr=l_r, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    model = train_model(model, criterion, optimizer,images_datasets, exp_lr_scheduler,
                        device, epochs)
    model.class_to_idx = images_datasets['train'].class_to_idx
    checkpoint = {'class_to_idx': model.class_to_idx,
              'arch': in_arg.arch,
              'classifier': model.classifier,
              'state_dict':model.state_dict(),
              'epochs': in_arg.epochs,
              'learning_rate': in_arg.learning_rate,
              'hidden_units': in_arg.hidden_units
              }
    torch.save(checkpoint, in_arg.save_dir)
    end_time = time()


def get_input_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('-d','--data_dir',type=str,default='flowers',help='The folder of the training images')
    parser.add_argument('-s','--save_dir',type=str,default='checkpoint.pth',help='The name of directory and file to save checkpoint')
    parser.add_argument('-a','--arch',type=str,default='vgg16',choices=['vgg16','alexnet'],help='Choose model: vgg16 or alexnet')
    parser.add_argument('-l','--learning_rate',type=float,default=0.01,help='Define learning rate')
    parser.add_argument('-u','--hidden_units',type=int,default=2000,help='Define hidden units between 4096 and 102')
    parser.add_argument('-e','--epochs',type=int,default=6,help='Define number of epochs')
    parser.add_argument('--gpu',type=bool,default=True,help='Choose GPU or CPU')    
    return parser.parse_args()
    
if __name__ == "__main__":
    main()     
