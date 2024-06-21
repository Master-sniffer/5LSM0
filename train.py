"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
from model import Model
from torchvision.datasets import Cityscapes
import argparse

try:
    from torchmetrics.classification import MulticlassJaccardIndex
except Exception as exc:
    print(exc)

        
import matplotlib.pyplot as plt
import numpy as np

import os
import numpy as np
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import wandb
import torch.optim.lr_scheduler as lr_scheduler
import random

import torchvision.transforms.v2 as transforms


import gc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

wandb.login(key='2320d550e058549752580c5d1c7862e3413ee44b',relogin=True)

wandb.init(
    project="my-awesome-project",
    magic=True,
    anonymous="allow",
    force=True
    

) 


try:
    print("here")
    import utils
except Exception:
    print("or here")
    import sys
    sys.path.insert(1, '/kaggle/input/5lsm0-neural-networks-for-cv-dataset')
    import utils


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=".", help="Path to the data")

    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def generate_random_colormap(num_classes):
    colormap = {}
    for i in range(num_classes):
        colormap[i] = tuple(random.randint(0, 255) for _ in range(3))
    return colormap

# Then, use the colormap as before
def colorize_mask(mask,colormap):
    """Convert a label mask to an RGB image."""
    mask_colorized = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in colormap.items():
        mask_colorized[mask == label] = color
    return mask_colorized

def visualize_images_and_masks(loader):
    dataiter = iter(loader)
    images, masks = dataiter.__next__()

    fig = plt.figure(figsize=(16, 10))
    for idx in np.arange(4):
        ax1 = fig.add_subplot(2, 4, 2*idx+1, xticks=[], yticks=[])
        ax2 = fig.add_subplot(2, 4, 2*idx+2, xticks=[], yticks=[])

        plt.sca(ax1)
        plt.imshow(np.transpose(images[idx].cpu(), (1, 2, 0)))
        if idx == 0:
            ax1.set_title('Images')

        plt.sca(ax2)
        plt.imshow(masks[idx].squeeze().cpu(), cmap="gray")
        if idx == 0:
            ax2.set_title('Masks')
    plt.show()


import torch.nn.functional as F

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight 
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255)

    def forward(self, input, target):

        ce_loss = self.cross_entropy(input,target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""
    print("Starting the main method...")

    
    """define your model, trainingsloop optimitzer etc. here"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # You don't have a gpu, so use cpu

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from torchvision import datasets
    import torchvision.transforms.v2 as transforms
    from torchvision.datasets import Cityscapes
    from argparse import ArgumentParser
    import wandb
    import torch.optim.lr_scheduler as lr_scheduler
    import random

    
    
    transform_data = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ColorJitter(contrast=0.5),
    transforms.RandomRotation([-70,70]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 255.0),  # Scale the tensor values to the range [0, 1]

    ])

    # Apply transformations to the dataset
    try:
        transformed_dataset = Cityscapes("/kaggle/input/5lsm0-neural-networks-for-cv-dataset", split='train', mode='fine', target_type='semantic',transforms=transform_data)#, transform=transform_data, target_transform=transform_target)
    except Exception:
        transformed_dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic',transforms=transform_data)

    
    try:
        transformed_test_dataset = Cityscapes("/kaggle/input/5lsm0-neural-networks-for-cv-dataset", split='test', mode='fine', target_type='semantic',transforms= transform_data)#, transform=transform_data, target_transform=transform_target)
        val_loader =  DataLoader(transformed_test_dataset, batch_size=128, shuffle=False, num_workers=18)
    except Exception:
        val_loader=None
        
    # Define the split points
    split_point1 = int(0.7 * len(transformed_dataset))  # 70% for training
    split_point2 = int(0.8 * len(transformed_dataset))  # 20% for validation, 10% for testing

    # Split the dataset
    transformed_train_dataset = torch.utils.data.Subset(transformed_dataset, range(split_point1))
    transformed_val_dataset = torch.utils.data.Subset(transformed_dataset, range(split_point1, split_point2))
    transformed_test = torch.utils.data.Subset(transformed_dataset, range(split_point2, len(transformed_dataset)))


    bs = 32
    print("Batch size ", bs)

    # Pass the transformed dataset to the DataLoader
    train_loader = DataLoader(transformed_train_dataset, batch_size=bs, shuffle=True, num_workers=18)
    val_loader = DataLoader(transformed_val_dataset, batch_size=bs, shuffle=True, num_workers=18)
    test_loader = DataLoader(transformed_test, batch_size=bs, shuffle=False, num_workers=18)

    encoder_cfg = [3, 64, 128, 256, 512]  # Example encoder configuration
    bottleneck_cfg = [512, 1024]  # Example bottleneck configuration
    decoder_cfg = [1024, 512, 256, 128, 64]  # Example decoder configuration


    output_channels = 19  # Number of output channels for segmentation

 
    try:
        print(encoder_cfg)

        model=Model(encoder_cfg, bottleneck_cfg, decoder_cfg, output_channels).cuda()
        print("Using CUDA")
        
    except Exception:
        model=Model()

        print("Using CPU")

        
    num_epochs = 200
    print(num_epochs)
    learning_rate = 0.001
    print("USING ", learning_rate)
    
    best_train_loss= 100
    best_test_loss= 100

    # Set the loss function and optimizer
    from torch import tensor
    from torchmetrics.classification import Dice
    from torchmetrics.functional.classification import dice
    criterion = FocalLoss() 
    print(criterion)
    wd = 0.001 
    print("WD: ", wd)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=wd,amsgrad=True)
    
    print("step_size", 20)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    print(scheduler.get_last_lr(), " LEARNING RATE")
    count = 0
    loss_list = []
    iteration_list = []


    early_stopper = EarlyStopper(patience=15, min_delta=10)

    import random, shutil, os
    num = random.randint(1,1000)
    os.mkdir(str(num))    
    print("RANDOM: ",num)

    # Train the model
    for epoch in range(num_epochs): 
        running_loss = 0.0
        test_loss=0.0
        model.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), (data[1]*255).to(device).long()
            optimizer.zero_grad()

            outputs = model(inputs)
            labels = labels.squeeze(1)  # remove the extra dimension
            labels = utils.map_id_to_train_id(labels).to(device)
            print(outputs.shape)
            print(labels.unique())
            loss = criterion(outputs, labels)
            v=epoch + 1
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            wandb.log({'loss': running_loss/(i+1)})
            wandb.log({'train_Iteration': i})
            wandb.log({'Train_Epoch':float(v)})

            print(f'Epoch {epoch + 1}, Iteration [{i}/{len(train_loader)}], Loss: {running_loss/(i+1)}, Model is in mode training: {model.training}',outputs.shape,inputs.shape)

            
        
        if((running_loss / len(train_loader))<best_train_loss):
            best_train_loss=running_loss / len(train_loader)
            print(f"Updated train loss, now it is {best_train_loss}")
        print(f'Finished epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

        with torch.no_grad():
            model.eval()
            
            for i, data in enumerate(test_loader):
                print()

                inputs, labels = data[0].to(device), (data[1]*255).to(device).long()
                labels=labels.squeeze(1)
                labels = utils.map_id_to_train_id(labels).to(device)
                outputs = model(inputs)


                loss = criterion(outputs, labels)
                v=epoch + 1


                test_loss += loss.item()
                wandb.log({'test_loss': running_loss/(i+1)})
                wandb.log({'test_Iteration': i})
                wandb.log({'TEST_Epoch':float(v)})

                print(f'TEST: Epoch {epoch + 1}, Iteration [{i}/{len(test_loader)}], Loss: {running_loss/(i+1)}, Model is in mode training: {model.training}') #,outputs.shape,inputs.shape
        
            if((test_loss / len(test_loader))<best_test_loss):
                best_test_loss=test_loss / len(test_loader)
                print(f"Updated test loss, now it is {best_test_loss}")
        
        scheduler.step()
        print(scheduler.get_last_lr(), " NEW LEARNING RATE")
        print(f'Finished TEST epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(test_loader):.4f}')

        
        def visual_preds(model, inputs,labels,outputs, epoch):
            # Visualize the first image in the batch
            model.eval()
            predicted = torch.argmax(outputs,dim=1)
            print(predicted.unique())
            img = inputs.cpu().numpy()[0]
            img = np.transpose(img, (1, 2, 0))
            img = np.clip(img, 0, 1)

            label = labels.cpu().numpy()[0]
            pred = predicted.cpu().numpy()[0]

            print("Ground truth label range:", np.min(label), np.max(label))
            print("Predicted label range:", np.min(pred), np.max(pred)) 
            
            num_classes = len(np.unique(label))  # Change this to the number of classes in your dataset
            colormap = generate_random_colormap(num_classes)

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img)
            axs[0].set_title('Input Image')
            axs[1].imshow(colorize_mask(label,colormap))
            axs[1].set_title('Ground Truth')
            axs[2].imshow(colorize_mask(pred,colormap))
            axs[2].set_title('Prediction')

            plt.show()
            plt.savefig(f'{num}/{epoch}_image.png')
            
            model.train()

        visual_preds(model,inputs,labels,outputs, epoch)

        if early_stopper.early_stop(test_loss):             
            break
  
    if (val_loader):
        with torch.no_grad():
                model.eval()
                running_loss = 0

                for i, data in enumerate(val_loader):
                    inputs, labels = data[0].to(device), (data[1]*255).to(device).long()
                    labels=labels.squeeze(1)

                    outputs = model(inputs)

                    loss = criterion(outputs, labels)

                    running_loss += loss.item()
                    wandb.log({'VAL_loss': running_loss/(i+1)})
                    wandb.log({'VAL_Iteration': i})


                    print(f'VAL: Iteration [{i}/{len(val_loader)}], Loss: {running_loss/(i+1)}',outputs.shape,inputs.shape)
    # save model
    print(f"Best losses --> Train:{best_train_loss} and Test:{best_test_loss}")
    torch.save(model.state_dict(), f"{num}/model_scripted.pth")

    # visualize some results

    pass



if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
