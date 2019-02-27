import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from collections import OrderedDict
from torch.optim import lr_scheduler
import os
import time
import copy
from PIL import Image
import json

def device_type(gpu=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def data_loaders(data_dir):
    #data_dir =  in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'valid','test']}
    
    return image_datasets

def cat_names(filepath):
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
    

   
def train_model(model, criterion, optimizer, image_datasets, scheduler, device, num_epochs):
    since = time.time()
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                  shuffle=True)
                   for x in ['train', 'valid','test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid','test']}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def category_names(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return class_names
        

def process_image(image_path):
    im=image_resize(image_path)
    #cropping
    width,height=im.size
    crop_size=224
    x=(width-crop_size)/2
    y=(height-crop_size)/2
    right=width-x
    bottom=height-y
    
    im=im.crop((x,y,right,bottom))
    
    #making color channels between 0-1
    np_image=np.array(im)
    
    mean=np.array([0.485,0.456,0.406])
    std=np.array([0.229,0.224,0.225])
    np_image=(((np_image/255)-mean)/std)
    #changing color channels order
    np_image=np_image.transpose((2,0,1))
    
    return np_image

def image_resize(image_path):
    im=Image.open(image_path)
    width,height=im.size
    if width<height:
        x=256/width
        new_height=height*x
        size=256,new_height
        im.thumbnail(size)
        
    else:
        y=256/height
        new_width=width*y
        size=new_width,256
        im.thumbnail(size)
        
    return im

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    if title:
        plt.title(title)    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    model.to(device)
    model.eval()
    image=process_image(image_path)
    torch_image= torch.from_numpy(image).float()
    torch_image.unsqueeze_(0)
    torch_image.requires_grad_(False)
    output=model(torch_image.cuda())
    probs, classes = torch.topk((F.softmax(output, dim=1)),topk,sorted=True)
    idx_to_class={v:k for k,v in model.class_to_idx.items()}   
    return [prob.item() for prob in probs[0].data], [idx_to_class[ix.item()] for ix in classes[0].data]


    