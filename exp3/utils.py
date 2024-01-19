import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def train(train_loader, optimizer, criterion, model, device):

    model.train()
    print('Training')

    running_loss = 0.0

    counter = 0

    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):

        counter += 1

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    
    epoch_loss = running_loss / counter

    return epoch_loss


def validation(val_loader, criterion, model, device):
    model.eval()

    print("Validation")

    running_loss = 0
    p = 0
    r = 0
    f1 = 0
    p_classes, r_classes, f1_classes = [0]*10, [0]*10, [0]*10

    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):

            counter += 1

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs)

            for class_id in range(10):
                if precision(outputs, labels, class_id) is not None:
                    p_classes[class_id] += precision(outputs, labels, class_id)
                else:
                    p_classes[class_id] += 0
                
                if recall(outputs, labels, class_id) is not None:
                    r_classes[class_id] += recall(outputs, labels, class_id)
                else:
                    r_classes[class_id] += 0

                if f1_score(outputs, labels, class_id) is not None:
                    f1_classes[class_id] += f1_score(outputs, labels, class_id)
                else:
                    f1_classes[class_id] += 0

            loss = criterion(outputs, labels)

            # print statistics
            running_loss += loss.item()
        
        epoch_loss = running_loss / counter
        p = sum(p_classes) / (len(p_classes) * counter)
        r = sum(r_classes) / (len(r_classes) * counter)
        f1 = sum(f1_classes) / (len(f1_classes) * counter)
        for class_id in range(10):
            p_classes[class_id] = p_classes[class_id] / counter
            r_classes[class_id] = r_classes[class_id] / counter
            f1_classes[class_id] = f1_classes[class_id] / counter

    return epoch_loss, p, r, f1, p_classes, r_classes, f1_classes


def plot_metrics(precision, recall, f1, x, y, save_path):
    plt.style.use("ggplot")

    plt.plot(precision, label='precision')
    plt.plot(recall, label='recall')
    plt.plot(f1, label='f1')
    plt.legend()
    plt.xlabel(x)
    plt.ylabel(y)

    plt.savefig(save_path, bbox_inches='tight')

    plt.clf()


def plot_class_metrics(precision, recall, f1, classname, x, y, save_path):
    plt.style.use("ggplot")

    plt.plot(precision, label=f'{classname}_precision')
    plt.plot(recall, label=f'{classname}_recall')
    plt.plot(f1, label=f'{classname}_f1')
    plt.legend()
    plt.xlabel(x)
    plt.ylabel(y)

    plt.savefig(save_path, bbox_inches='tight')

    plt.clf()


def plot_loss(train_loss, val_loss, x, y, save_path):
    plt.style.use("ggplot")

    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend()
    plt.xlabel(x)
    plt.ylabel(y)

    plt.savefig(save_path, bbox_inches='tight')

    plt.clf()


def precision(outputs, targets, label=None):
    predicted_labels = torch.argmax(outputs, dim=1)
    if label is not None:
        predicted_labels = predicted_labels
        targets = targets
        false_positives = torch.logical_and(predicted_labels == label, targets != label).sum().item()
        true_positives = (predicted_labels == targets).sum().item()

        if true_positives + false_positives != 0:
            return true_positives / (true_positives + false_positives)
        else:
            return None
    else:
        return None


def recall(outputs, targets, label=None):
    predicted_labels = torch.argmax(outputs, dim=1)
    if label is not None:
        predicted_labels = predicted_labels
        targets = targets
        false_negatives = torch.logical_and(predicted_labels != label, targets == label).sum().item()
        true_positives = (predicted_labels == targets).sum().item()

        if true_positives + false_negatives != 0:
            return true_positives / (true_positives + false_negatives)
        else:
            return None
    else:
        return None


def f1_score(outputs, targets, label=None):
    precision_value = precision(outputs, targets, label)
    recall_value = recall(outputs, targets, label)
    if precision_value is not None and recall_value is not None:
        return 2 * (precision_value * recall_value) / (precision_value + recall_value)
    else:
        return None

