from torchvision import models, datasets, transforms
import torch
from torch import optim
from torch import nn
from PIL import Image
import numpy as np
import argparse
import json






parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=str, help="Path to image")
parser.add_argument("checkpoint", type=str, help="checkpoint file")
parser.add_argument("--top_k" , type=int, default=1)
parser.add_argument("--category_names", type=str, default="cat_to_name.json")
parser.add_argument("--gpu", type=str, default='gpu')

args = parser.parse_args()

if 'vgg11' == args.checkpoint.split('_')[0]:
    model = models.vgg11(pretrained=True)
    classifier = nn.Sequential(nn.Linear(int(args.checkpoint.split('_')[1]), 4096),
                              nn.ReLU(),
                              nn.Linear(4096, 1024),
                              nn.ReLU(),
                              nn.Linear(1024, 512),
                              nn.ReLU(), 
                              nn.Linear(512, 102),
                              nn.LogSoftmax(dim=1)
                              )
    model.classifier = classifier
elif 'alexnet' == args.checkpoint.split('_')[0]:
    model = models.alexnet(pretrained=True)
    classifier = nn.Sequential(nn.Linear(int(args.checkpoint.split('_')[1]), 4096),
                              nn.ReLU(),
                              nn.Linear(4096, 1024),
                              nn.ReLU(),
                              nn.Linear(1024, 512),
                              nn.ReLU(), 
                              nn.Linear(512, 102),
                              nn.LogSoftmax(dim=1)
                              )
    model.classifier = classifier
    
train_dir = 'ImageClassifier/flowers/train'
data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
test_transform = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
image_datasets = datasets.ImageFolder(train_dir, transform = data_transforms)


# TODO: Using the image datasets and the trainforms, define the dataloaders


with open('ImageClassifier/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
   


    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)
    
    # Resize image with shortest side of 256 pixels
    size = 256
    width, height = image.size
    if width < height:
        new_width = size
        new_height = int(size * height / width)
    else:
        new_height = size
        new_width = int(size * width / height)
    image = image.resize((new_width, new_height))

    
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))


    np_image = np.array(image) / 255.0

# Normalize the image with mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions from (224, 224, 3) to (3, 224, 224)
    np_image = np_image.transpose((2, 0, 1))

    
    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    return tensor_image



def predict(image_path, model, topk=args.top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.float()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model.forward(image.to(device))
        ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)
    return top_p, top_class
    


def loading(file):
    checkpoint =  torch.load(file)
    
    new_model = model
    
    new_model.load_state_dict(checkpoint)
    new_model.eval()
    return new_model



loaded_checkpoint = loading(args.checkpoint)
model = loaded_checkpoint

print(loaded_checkpoint)

image_path = args.image_path


model.class_to_idx = image_datasets.class_to_idx


top_p, top_class = predict(image_path, model)

top_class= top_class.cpu().numpy()
top_p = top_p.cpu().numpy()

idx_to_class = {val: key for key, val in model.class_to_idx.items()}


lis = []
for i in range(args.top_k):
    
    one_class = idx_to_class[top_class[0, i]]
    lis.append(one_class)

flower_label = [cat_to_name[i] for i in lis]

for index, flower in enumerate(flower_label):
    print(f"No {index + 1} prediction is {flower}..........probability is {top_p[0][index]}")