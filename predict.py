import argparse
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import datasets,transforms,models
import os
import matplotlib.pyplot as plt
from functions import cat_names,data_loaders, process_image,image_resize,predict

def main():
    in_arg = get_input_args()
    images_data=data_loaders(in_arg.data_dir)
    if in_arg.gpu is True:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            print("Using CPU since GPU is not available")
    model=load_checkpoint(in_arg.checkpoint)
    cat_to_name=cat_names(in_arg.cat_name)
    probs, classes = predict(in_arg.image, model,device,in_arg.top_k)
    #df = dict(sorted(zip(names, probs)))
    names = [cat_to_name[name] for name in classes]
    print(probs, [cat_to_name[name] for name in classes])
    p=probs.index(max(probs))
    print('Predicted flower in image is {} with a probability of {}'.format(names[p],probs[p]))
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict=checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def get_input_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('-d','--data_dir',type=str,default='flowers',help='The folder of the training images')
    parser.add_argument('--image',type=str,default='flowers/test/42/image_05696.jpg',help='The image to predict')
    parser.add_argument('--checkpoint',type=str,default='checkpoint.pth',help='The saved model training checkpoint')
    parser.add_argument('--top_k',type=int,default=5,help='Number of top classes predictions')
    parser.add_argument('--cat_name',type=str,default='cat_to_name.json',help='The JSON file with the cats names')
    parser.add_argument('--gpu',type=bool,default=True,help='Choose GPU or CPU')    
    return parser.parse_args()    
    
    
if __name__ == "__main__":
    main()   