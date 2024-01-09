import torch
from torchvision.models.detection.mask_rcnn import resnet50
import torch.optim as optim
import torch.nn as nn
import json

from utils import train, validation, plot_loss, plot_metrics, plot_class_metrics
from data_utils import get_dataloaders

import datetime
import os
import wandb
import numpy as np 

# open config file
with open('config.json') as f:
    config = json.load(f) 


batch_size = config["batch_size"]
epochs = config["epochs"]
model_names = config["model"]
learning_rate = config["lr"]
train_loader, val_loader, _ = get_dataloaders(batch_size)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
config["n_classes"] = len(classes)
criterion = nn.CrossEntropyLoss()

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
config["device"] = device

model = resnet50().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

wandb.init(
    # set the wandb project where this run will be logged
    project="homework_1",
    # track hyperparameters and run metadata
    config=config
)

if __name__ == '__main__':
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    valid_precision, valid_recall, valid_f1 = [], [], []
    valid_precision_classes, valid_recall_classes, valid_f1_classes = [], [], []

    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss = train(
            train_loader, 
            optimizer, 
            criterion,
            model,
            device
        )
        valid_epoch_loss, p, r, f1, p_classes, r_classes, f1_classes = validation( 
            val_loader, 
            criterion,
            model,
            device
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)

        valid_precision.append(p)
        valid_recall.append(r)
        valid_f1.append(f1)

        valid_precision_classes.append(p_classes)
        valid_recall_classes.append(r_classes)
        valid_f1_classes.append(f1_classes)

        wandb.log({"train_loss": train_epoch_loss, "valid_loss": valid_epoch_loss})
        wandb.log({"precision": p, "recall": r, "f1_score": f1})
        for class_id in range(config["n_classes"]):
            wandb.log({f'val/{classes[class_id]}_precision': p_classes[class_id]})
            wandb.log({f'val/{classes[class_id]}_recall': r_classes[class_id]})
            wandb.log({f'val/{classes[class_id]}_f1': f1_classes[class_id]})

        print(f"Training loss: {train_epoch_loss:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}")
        print('-'*50)
        
    print('TRAINING COMPLETE')


result_path = os.path.join("./results/runs/", str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_"))
os.makedirs(result_path)
model_path = os.path.join(result_path, "model")
torch.save(model.state_dict(), model_path)

with open(os.path.join(result_path, "config.json"), "w") as f:
    json.dump(config , f)


plot_loss(train_loss, valid_loss, "epochs", "loss", os.path.join(result_path, "validation_loss"))
plot_metrics(valid_precision, valid_recall, valid_f1, "epochs", "validation_metrics", os.path.join(result_path, "validation_metrics"))

for class_id in range(config["n_classes"]):
    plot_class_metrics(np.array(valid_precision_classes).T.tolist()[class_id],
                       np.array(valid_recall_classes).T.tolist()[class_id],
                       np.array(valid_f1_classes).T.tolist()[class_id],
                       classes[class_id],
                       "epochs",
                       f"{classes[class_id]}_metrcis",
                       os.path.join(result_path, f"{classes[class_id]}_metrcis"))

wandb.finish()