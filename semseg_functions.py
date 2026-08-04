import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image
import tqdm
import pandas as pd
# import pytorch_lightning as pl
import os
import copy
from skimage.io import imread
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def show_filters(weights, n_cols=8, cmap='gray'):
    """
    Display convolutional filters as images.
    
    Args:
        weights (torch.Tensor): Conv layer weights of shape [out_channels, in_channels, H, W]
        n_cols (int): Number of columns in the display grid.
        cmap (str): Color map to use ('gray', 'viridis', etc.)
    """
    # Detach and move to CPU
    weights = weights.detach().cpu()

    # If multi-channel (e.g., 3 input channels), reduce to grayscale by averaging
    if weights.shape[1] == 3:
        weights = weights.mean(dim=1, keepdim=True)

    # Normalize each filter for better contrast
    weights = (weights - weights.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0])
    weights = weights / weights.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    n_filters = weights.shape[0]
    n_rows = int(np.ceil(n_filters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    axes = axes.flatten()

    for i in range(n_filters):
        filter_img = weights[i, 0]  # first input channel (or averaged)
        axes[i].imshow(filter_img, cmap=cmap)
        axes[i].axis('off')

    # Turn off remaining axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


# --- Utility to Register Hook ---
def capture_activation(model, layer_name, activations_dict):
    """
    Register a forward hook to capture activations from a given layer.
    """
    def hook_fn(module, input, output):
        activations_dict[layer_name] = output.detach().cpu()
    layer = dict([*model.named_modules()])[layer_name]
    return layer.register_forward_hook(hook_fn)

# --- Visualization Function ---
def plot_input_activation_output(input_img, activations, output_img, act_layer_name, n_cols=8):
    """
    Plot input image, activations, and model output side by side.
    """

    # Activations
    act = activations[act_layer_name].squeeze(0)  # shape: [C, H, W]
    n_features = act.shape[0]
    n_rows = int(np.ceil(n_features / n_cols))
    fig_act, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    axes = axes.flatten()
    for i in range(min(n_features, len(axes))):
        axes[i].imshow(act[i], cmap='viridis')
        axes[i].axis('off')
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.suptitle(f"Activations from {act_layer_name}", fontsize=14)

    plt.tight_layout()
    plt.show()

def get_layer_activations(model, layer_name, input_tensor):
    """
    Runs a forward pass and captures activations from a specified layer.

    Args:
        model (torch.nn.Module): The model to inspect.
        layer_name (str): Name of the layer to hook (e.g., 'encoder.conv1').
        input_tensor (torch.Tensor): Input image tensor of shape [1, C, H, W].

    Returns:
        output (torch.Tensor): Model output.
        activations (dict): Dictionary with one key (layer_name) -> activations tensor.
    """
    activations = {}

    def hook_fn(module, input, output):
        activations[layer_name] = output.detach().cpu()

    layer = dict([*model.named_modules()])[layer_name]
    hook = layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    hook.remove()
    return output, activations

def run_and_visualize_layer_activation(model, layer_name, input_tensor, n_cols=8, cmap='viridis'):
    """
    Runs a forward pass, captures activations from a specified layer, and visualizes
    input, activations, and output.

    Args:
        model (torch.nn.Module): The model to inspect.
        layer_name (str): Name of the layer to hook (e.g., 'encoder.conv1').
        input_tensor (torch.Tensor): Input tensor of shape [1, C, H, W].
        n_cols (int): Columns in activation grid.
        cmap (str): Color map for activations.
    """
    activations = {}

    def hook_fn(module, input, output):
        activations[layer_name] = output.detach().cpu()

    layer = dict([*model.named_modules()])[layer_name]
    hook = layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    hook.remove()

    # Visualize everything
    plot_input_activation_output(input_tensor.cpu(), activations, output, layer_name, n_cols=n_cols)

def load_imgs_labels(train_dir="./train",val_dir="./val"):
    train_imgs=np.stack(list(map(imread,sorted(glob.glob(os.path.join(train_dir,"imgs","*.png"))))))
    X_train=torch.FloatTensor(train_imgs).permute((0,3,1,2))/255
    
    val_imgs=np.stack(list(map(imread,sorted(glob.glob(os.path.join(val_dir,"imgs","*.png"))))))
    X_val=torch.FloatTensor(val_imgs).permute((0,3,1,2))/255
    
    train_lbls=np.stack(list(map(lambda x: imread(x)[...,0].astype(int),sorted(glob.glob(os.path.join(train_dir,"labels","*.png"))))))
    Y_train=torch.LongTensor(train_lbls)
    
    val_lbls=np.stack(list(map(lambda x: imread(x)[...,0].astype(int),sorted(glob.glob(os.path.join(val_dir,"labels","*.png"))))))
    Y_val=torch.LongTensor(val_lbls)
    
    return X_train,Y_train,X_val,Y_val

def load_model(path_dir = None, model_key="unet", encoder_name="resnet18", device="cpu",encoder_weights="imagenet"):
    model=dict(unet=smp.Unet,
                fpn=smp.FPN).get(model_key, smp.Unet)
    model=model(classes=3,in_channels=3, encoder_name=encoder_name, encoder_weights=encoder_weights)
    if path_dir is not None:
        model_fnames=glob.glob(path_dir + '/*_model.pkl')
        if len(model_fnames)>0:
            model_list=sorted(model_fnames, key=os.path.getmtime)
            model.load_state_dict(torch.load(model_list[-1], map_location="cpu"))
    return model

def train_model(train_dataset, val_dataset, model=None, save=True, n_epochs=30, model_key="unet", encoder_name="resnet18", path_dir="./seg_models", device="cpu"):
    
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Check if a pre-loaded model was supplied
    if model is None:
        encoder_name = "resnet18" if encoder_name not in smp.encoders.get_encoder_names() else encoder_name
        model = load_model(None, model_key, encoder_name, device)
    else:
        # Move loaded model to the active computing device (CPU or GPU)
        model = model.to(device)
        
    optimizer=torch.optim.Adam(model.parameters(), lr=5e-5)
    
    #learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
    )
    
    train_labels = train_dataset.y.numpy().flatten()
    class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight = torch.FloatTensor(class_weight).to(device)

    dice_class_weight = torch.sqrt(class_weight)
    ce_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight)
    dice_loss_fn = smp.losses.DiceLoss(mode='multiclass', classes=[0, 1, 2])
    if not os.path.exists(path_dir): os.makedirs(path_dir,exist_ok=True)
    min_loss=np.inf
    
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"./seg_models/urothelial_unet_{run_timestamp}_model.pkl"

    history = {"train_loss": [], "val_loss": []}
    best_model = copy.deepcopy(model.state_dict())
    for epoch in range(1,n_epochs+1):
        # training set 
        model.train(True)
        epoch_train_losses = [] # Track losses for each batch in this epoch
        for i,(x,y_true) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                x=x.to(device)
                y_true=y_true.to(device)
            optimizer.zero_grad()
            y_pred=model(x)
            loss = (0.4 * ce_loss_fn(y_pred, y_true)) + (0.6 * dice_loss_fn(y_pred, y_true))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
             # Save batch loss
            epoch_train_losses.append(loss.item())
            print(f"Training: Epoch {epoch}, Batch {i}, Loss: {round(loss.item(),3)}")
        #Compute and save the average training loss for this epoch
        avg_train_loss = np.mean(epoch_train_losses)
        history["train_loss"].append(avg_train_loss)
        # validation set
        model.train(False)
        with torch.no_grad():
            val_loss = []
            for i,(x,y_true) in enumerate(val_dataloader):
                if torch.cuda.is_available():
                    x=x.to(device)
                    y_true=y_true.to(device)
                y_pred=model(x)
                loss = (0.4 * ce_loss_fn(y_pred, y_true)) + (0.6 * dice_loss_fn(y_pred, y_true))
                val_loss.append(loss.item())
            
            val_loss=np.mean(val_loss)
             #Save the average validation loss for this epoch
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss) #changed here 7/17/26
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Val: Epoch {epoch}, Loss: {round(val_loss,3)} | LR: {current_lr}")
            
            if val_loss < min_loss:
                min_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                if save:
                    torch.save(model.state_dict(), filename)  # overwrites, no new timestamp
                    
    model.load_state_dict(best_model)
    return model, history

def make_predictions(X_val,model=None,save=True,path_dir = "./seg_models", model_key="unet", encoder_name="resnet18", device="cpu"):
    val_data=TensorDataset(X_val)
    val_dataloader=DataLoader(val_data,batch_size=8,shuffle=False)
    predictions=[]
    encoder_name="resnet18" if encoder_name not in smp.encoders.get_encoder_names() else encoder_name
    # load most recent saved model
    if model is None and save:
        model=load_model(path_dir, model_key, encoder_name, device)
    model=model.to(device)
    model.train(False)
    with torch.no_grad():
        for i,(x,) in enumerate(val_dataloader):
            if torch.cuda.is_available():
                x=x.to(device)
            y_pred=torch.softmax(model(x),1).detach().cpu().numpy()
            predictions.append(y_pred)
    predictions=np.concatenate(predictions,axis=0)
    return predictions