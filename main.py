import torch
from collections import Counter
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
import os
from glob import glob
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
#from torchsummary import summary
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import numpy as np
import tqdm
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data science tools
import numpy as np
import pandas as pd
import os

# Image manipulations
from PIL import Image

from timeit import default_timer as timer

# Visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
from torch import Tensor, nn
from torch.nn.functional import interpolate
def return_prediction(file_api):
    def imshow_tensor(image, ax=None, title=None):
        """Imshow for Tensor."""

        if ax is None:
            fig, ax = plt.subplots()

        # Set the color channel as the third dimension
        image = image.numpy().transpose((1, 2, 0))

        # Reverse the preprocessing steps
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean

        # Clip the image pixel values
        image = np.clip(image, 0, 1)

        ax.imshow(image)
        plt.axis('off')

        return ax, image

    traindir = f"data/train"
    validdir = f"data/val"
    testdir = f"data/test"
    
    checkpoint_path = f'resnet50-transfer.pth'

    # Change to fit hardware
    batch_size = 512

    # Whether to train on a gpu
    train_on_gpu = cuda.is_available()
    print(f'Train on gpu: {train_on_gpu}')

    multi_gpu = False
    # Number of gpus
    if train_on_gpu:
        gpu_count = cuda.device_count()
        print(f'{gpu_count} gpus detected.')
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    # Image transformations
    image_transforms = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # Imagenet standards
        ]),
        # Validation does not use augmentation
        'valid':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        
            # Test does not use augmentation
        'test':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    folder_path = './CarDamageTest/images/'
    ex_img = Image.open(folder_path + file_api)
        

    t = image_transforms['train']
    plt.figure(figsize=(24, 24))

    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        _ = imshow_tensor(t(ex_img), ax=ax)

    plt.tight_layout()

    # Datasets from folders
    data = {
        'train':
        datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
        'valid':
        datasets.ImageFolder(root=validdir, transform=image_transforms['valid']),
        'test':
        datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
    }

    # Dataloader iterators, make sure to shuffle
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True,num_workers=0),
        'val': DataLoader(data['valid'], batch_size=batch_size, shuffle=True,num_workers=0),
        'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True,num_workers=0)
    }

    # Iterate through the dataloader once
    trainiter = iter(dataloaders['train'])
    features, labels = next(trainiter)
    features.shape, labels.shape

    categories = []
    for d in os.listdir(traindir):
        categories.append(d)
        
    n_classes = len(categories)
    print(f'There are {n_classes} different classes.')
    print(f'{categories}')

    class_to_idx = data['train'].class_to_idx
    idx_to_class = {
        idx: class_
        for class_, idx in data['train'].class_to_idx.items()
    }

    train_cnts = Counter([idx_to_class[x] for x in data['train'].targets])
    val_cnts = Counter([idx_to_class[x] for x in data['valid'].targets])
    test_cnts = Counter([idx_to_class[x] for x in data['test'].targets])
    train_cnts = pd.DataFrame({'cat' :list(train_cnts.keys()), 'train_cnt': list(train_cnts.values())})
    val_cnts = pd.DataFrame({'cat' :list(val_cnts.keys()), 'val_cnt': list(val_cnts.values())})
    test_cnts = pd.DataFrame({'cat' :list(test_cnts.keys()), 'test_cnt': list(test_cnts.values())})
    cnt_df = pd.merge(train_cnts,val_cnts,on='cat',how='left').merge(test_cnts,on='cat',how='left')
    cnt_df.head()

    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False
    print(model)

    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
                          nn.Linear(n_inputs, 256), 
                          nn.ReLU(), 
                          nn.Dropout(0.4),
                          nn.Linear(256, n_classes),                   
                          nn.LogSoftmax(dim=1))
    model.fc

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    if train_on_gpu:
        model = model.to('cuda')

    if multi_gpu:
        model = nn.DataParallel(model)
    if multi_gpu:
        print(model.module.fc)
    else:
        print(model.fc)

    model.class_to_idx = data['train'].class_to_idx
    model.idx_to_class = {
        idx: class_
        for class_, idx in model.class_to_idx.items()
    }

    print(list(model.idx_to_class.items())[:10])


    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())
    for p in optimizer.param_groups[0]['params']:
        if p.requires_grad:
            print(p.shape)



    def train(model,
              criterion,
              optimizer,
              train_loader,
              valid_loader,
              save_file_name,
              max_epochs_stop=3,
              n_epochs=20,
              print_every=1):
        """Train a PyTorch Model

        Params
        --------
            model (PyTorch model): cnn to train
            criterion (PyTorch loss): objective to minimize
            optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
            train_loader (PyTorch dataloader): training dataloader to iterate through
            valid_loader (PyTorch dataloader): validation dataloader used for early stopping
            save_file_name (str ending in '.pt'): file path to save the model state dict
            max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
            n_epochs (int): maximum number of training epochs
            print_every (int): frequency of epochs to print training stats

        Returns
        --------
            model (PyTorch model): trained cnn with best weights
            history (DataFrame): history of train and validation loss and accuracy
        """

        # Early stopping intialization
        epochs_no_improve = 0
        valid_loss_min = np.Inf

        valid_max_acc = 0
        history = []

        # Number of epochs already trained (if using loaded in model weights)
        try:
            print(f'Model has been trained for: {model.epochs} epochs.\n')
        except:
            model.epochs = 0
            print(f'Starting Training from Scratch.\n')

        overall_start = timer()

        # Main loop
        for epoch in range(n_epochs):

            # keep track of training and validation loss each epoch
            train_loss = 0.0
            valid_loss = 0.0

            train_acc = 0
            valid_acc = 0

            # Set to training
            model.train()
            start = timer()

            # Training loop
            for ii, (data, target) in enumerate(train_loader):
                # Tensors to gpu
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()

                # Clear gradients
                optimizer.zero_grad()
                # Predicted outputs are log probabilities
                output = model(data)

                # Loss and backpropagation of gradients
                loss = criterion(output, target)
                loss.backward()

                # Update the parameters
                optimizer.step()

                # Track train loss by multiplying average loss by number of examples in batch
                train_loss += loss.item() * data.size(0)

                # Calculate accuracy by finding max log probability
                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                # Need to convert correct tensor from int to float to average
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                # Multiply average accuracy times the number of examples in batch
                train_acc += accuracy.item() * data.size(0)

                # Track training progress
                print(
                    f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                    end='\r')

            # After training loops ends, start validation
            else:
                model.epochs += 1

                # Don't need to keep track of gradients
                with torch.no_grad():
                    # Set to evaluation mode
                    model.eval()

                    # Validation loop
                    for data, target in valid_loader:
                        # Tensors to gpu
                        if train_on_gpu:
                            data, target = data.cuda(), target.cuda()

                        # Forward pass
                        output = model(data)

                        # Validation loss
                        loss = criterion(output, target)
                        # Multiply average loss times the number of examples in batch
                        valid_loss += loss.item() * data.size(0)

                        # Calculate validation accuracy
                        _, pred = torch.max(output, dim=1)
                        correct_tensor = pred.eq(target.data.view_as(pred))
                        accuracy = torch.mean(
                            correct_tensor.type(torch.FloatTensor))
                        # Multiply average accuracy times the number of examples
                        valid_acc += accuracy.item() * data.size(0)

                    # Calculate average losses
                    train_loss = train_loss / len(train_loader.dataset)
                    valid_loss = valid_loss / len(valid_loader.dataset)

                    # Calculate average accuracy
                    train_acc = train_acc / len(train_loader.dataset)
                    valid_acc = valid_acc / len(valid_loader.dataset)

                    history.append([train_loss, valid_loss, train_acc, valid_acc])

                    # Print training and validation results
                    if (epoch + 1) % print_every == 0:
                        print(
                            f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                        )
                        print(
                            f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                        )

                    # Save the model if validation loss decreases
                    if valid_loss < valid_loss_min:
                        # Save model
                        torch.save(model.state_dict(), save_file_name)
                        # Track improvement
                        epochs_no_improve = 0
                        valid_loss_min = valid_loss
                        valid_best_acc = valid_acc
                        best_epoch = epoch

                    # Otherwise increment count of epochs with no improvement
                    else:
                        epochs_no_improve += 1
                        # Trigger early stopping
                        if epochs_no_improve >= max_epochs_stop:
                            print(
                                f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                            )
                            total_time = timer() - overall_start
                            print(
                                f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                            )

                            # Load the best state dict
                            model.load_state_dict(torch.load(save_file_name))
                            # Attach the optimizer
                            model.optimizer = optimizer

                            # Format history
                            history = pd.DataFrame(
                                history,
                                columns=[
                                    'train_loss', 'valid_loss', 'train_acc',
                                    'valid_acc'
                                ])
                            return model, history

        # Attach the optimizer
        model.optimizer = optimizer
        # Record overall time and print out stats
        total_time = timer() - overall_start
        print(
            f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
        )
        print(
            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
        )
        # Format history
        history = pd.DataFrame(
            history,
            columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
        return model, history





    def save_checkpoint(model, path):
        """Save a PyTorch model checkpoint

        Params
        --------
            model (PyTorch model): model to save
            path (str): location to save model. Must start with `model_name-` and end in '.pth'

        Returns
        --------
            None, save the `model` to `path`

        """

        model_name = path.split('-')[0]
        assert (model_name in ['vgg16', 'resnet50'
                               ]), "Path must have the correct model name"

        # Basic details
        checkpoint = {
            'class_to_idx': model.class_to_idx,
            'idx_to_class': model.idx_to_class,
            'epochs': model.epochs,
        }

        # Extract the final classifier and the state dictionary
        if model_name == 'vgg16':
            # Check to see if model was parallelized
            if multi_gpu:
                checkpoint['classifier'] = model.module.classifier
                checkpoint['state_dict'] = model.module.state_dict()
            else:
                checkpoint['classifier'] = model.classifier
                checkpoint['state_dict'] = model.state_dict()

        elif model_name == 'resnet50':
            if multi_gpu:
                checkpoint['fc'] = model.module.fc
                checkpoint['state_dict'] = model.module.state_dict()
            else:
                checkpoint['fc'] = model.fc
                checkpoint['state_dict'] = model.state_dict()

        # Add the optimizer
        checkpoint['optimizer'] = model.optimizer
        checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

        # Save the data to the path
        torch.save(checkpoint, path)




    def load_checkpoint(path):
        """Load a PyTorch model checkpoint

        Params
        --------
            path (str): saved model checkpoint. Must start with `model_name-` and end in '.pth'

        Returns
        --------
            None, save the `model` to `path`

        """

        # Get the model name
        model_name = path.split('-')[0]
        assert (model_name in ['vgg16', 'resnet50'
                               ]), "Path must have the correct model name"

        # Load in checkpoint
        checkpoint = torch.load(path, map_location=torch.device('cpu'))

        if model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            # Make sure to set parameters as not trainable
            for param in model.parameters():
                param.requires_grad = False
            model.classifier = checkpoint['classifier']

        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            # Make sure to set parameters as not trainable
            for param in model.parameters():
                param.requires_grad = False
            model.fc = checkpoint['fc']

        # Load in the state dict
        model.load_state_dict(checkpoint['state_dict'])

        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} total gradient parameters.')

        # Move to gpu
        if multi_gpu:
            model = nn.DataParallel(model)

        if train_on_gpu:
            model = model.to('cuda')

        # Model basics
        model.class_to_idx = checkpoint['class_to_idx']
        model.idx_to_class = checkpoint['idx_to_class']
        model.epochs = checkpoint['epochs']

        # Optimizer
        optimizer = checkpoint['optimizer']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer

    model, optimizer = load_checkpoint(path=checkpoint_path)


    def process_image(image_path):
        """Process an image path into a PyTorch tensor"""

        image = Image.open(image_path)
        # Resize
        img = image.resize((256, 256))

        # Center crop
        width = 256
        height = 256
        new_width = 224
        new_height = 224

        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        img = img.crop((left, top, right, bottom))

        # Convert to numpy, transpose color dimension and normalize
        img = np.array(img).transpose((2, 0, 1)) / 256

        # Standardization
        means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

        img = img - means
        img = img / stds

        img_tensor = torch.Tensor(img)

        return img_tensor
    def predict(image_path, model, topk=5):
        """Make a prediction for an image using a trained model

        Params
        --------
            image_path (str): filename of the image
            model (PyTorch model): trained model for inference
            topk (int): number of top predictions to return

        Returns
            
        """
        real_class = image_path.split('/')[-2]

        # Convert to pytorch tensor
        img_tensor = process_image(image_path)

        # Resize
        if train_on_gpu:
            img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
        else:
            img_tensor = img_tensor.view(1, 3, 224, 224)

        # Set to evaluation
        with torch.no_grad():
            model.eval()
            # Model outputs log probabilities
            out = model(img_tensor)
            ps = torch.exp(out)

            # Find the topk predictions
            topk, topclass = ps.topk(topk, dim=1)

            # Extract the actual classes and probabilities
            top_classes = [
                model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
            ]
            top_p = topk.cpu().numpy()[0]

            return img_tensor.cpu().squeeze(), top_p, top_classes, real_class
    np.random.seed = 100


    def random_test_image():
        """Pick a random test image from the test directory"""
        root = "CarDamageTest/images/"
        img_path = root + np.random.choice(os.listdir(root))
        return img_path


    _ = imshow_tensor(process_image(random_test_image()))

    img, top_p, top_classes, real_class = predict(random_test_image(), model,topk=2)
    top_p, top_classes, real_class

    def display_prediction(image_path, model, topk):
        """Display image and preditions from model"""

        # Get predictions
        img, ps, classes, y_obs = predict(image_path, model, topk)
        # Convert results to dataframe for plotting
        result = pd.DataFrame({'p': ps}, index=classes)
        result = result['p']
        out = ps[0], classes[0]
        print(str(out))

        # Show the image
        plt.figure(figsize=(16, 5))
        ax = plt.subplot(1, 2, 1)
        ax, img = imshow_tensor(img, ax=ax)

        # Set title to be the actual class
        ax.set_title(y_obs, size=20)

        ax = plt.subplot(1, 2, 2)
        # Plot a bar plot of predictions
        print(image_path)

        value = []
        for (i, class_name) in enumerate(classes):
            value.append("{" + class_name + "}:{" + str(result[i]) + "}")
        return str(value), out

    return display_prediction(random_test_image(), model, topk=3)





from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import uvicorn
import os
from fastapi.responses import FileResponse

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




app = FastAPI()

templates = Jinja2Templates(directory="")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/CarDamageTest/images/")
async def upload(uploaded_file: UploadFile = File(...)):
    file_location = f"./CarDamageTest/images/{uploaded_file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())
    file = os.listdir("./CarDamageTest/images")
    print(file)
    file = file[0]
    file_api = file
    value, output = return_prediction(file_api)
   
    os.system(f"rm -rf ./CarDamageTest/images/{file}")
    return {"message": f"{output}"}
    
    


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port="8000")
