from torchvision import models, datasets, transforms
import torch
from torch import optim
from torch import nn
from PIL import Image
import numpy as np
import argparse
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("datadir", type=str, help="image path", default='ImageClassifier/flowers/train')
    parser.add_argument("--save_dir", type=str, help="save checkpoint", default="vgg11_25088")
    parser.add_argument("--arch", type=str, help="choose architecture", default='vgg11')
    parser.add_argument("--lr", type=int, help="hyperparameter", default=0.002)
    parser.add_argument("--hidden_units", type=int, help="hyperparameter", default=25088)
    parser.add_argument("--epochs", type=int, help="hyperparameter", default = 8)
    parser.add_argument("--gpu", type=str, help="hyperparameter", default='gpu')


    args = parser.parse_args()


    train_dir = args.datadir
    valid_dir = 'ImageClassifier/flowers/valid'
    test_dir = 'ImageClassifier/flowers/test'

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
    valid_datasets = datasets.ImageFolder(valid_dir, transform=test_transform)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transform)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=32, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True)





    vgg11=  models.vgg11(pretrained=True)
    alexnet = models.alexnet(pretrained=True)

    models = {'alexnet':alexnet, 'vgg11': vgg11}


    print("*******Model Details*****************")
    model = models[args.arch]
    print(model)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(args.hidden_units, 4096),
                              nn.ReLU(),
                              nn.Linear(4096, 1024),
                              nn.ReLU(),
                              nn.Linear(1024, 512),
                              nn.ReLU(), 
                              nn.Linear(512, 102),
                              nn.LogSoftmax(dim=1)
                              )
    model.classifier = classifier
    model.classifier



    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    epochs = args.epochs
    step = 0
    running_loss = 0


    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model.to(device);

    for e in range(epochs):
        for images, labels in train_dataloaders:
            step += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if step % 5 == 0:
                test_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for images, labels in valid_dataloaders:
                        images, labels = images.to(device), labels.to(device)
                        logits = model.forward(images)
                        batch_loss = criterion(logits, labels)

                        test_loss += batch_loss.item()
                        ps = torch.exp(logits)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {e + 1 }/ {epochs}...",
                      "Train Loss is {:.3f}.....".format(running_loss/5),
                     "Validation Loss is {:.3f}....".format(test_loss/len(train_dataloaders)),
                      "Validation accuracy: {:.3f}...".format(accuracy/len(valid_dataloaders)))
                running_loss = 0
                model.train()


                        # TODO: Do validation on the test set
    with torch.no_grad():
        test2_loss = 0
        accuracy = 0
        model.eval()

        for images, labels in test_dataloaders:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            loss = criterion(output, labels)
            test2_loss += loss.item()
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        print("Loss is :{:.3f}".format(test2_loss/len(test_dataloaders)),
            "Accuracy is : {:.3f}".format(accuracy/len(test_dataloaders)))


    torch.save(model.state_dict(), args.save_dir)

        






