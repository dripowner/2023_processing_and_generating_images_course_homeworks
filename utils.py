import torch
import tqdm


def train(autoencoder, data, device, epochs=20, vae=False):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in tqdm.tqdm(range(epochs)):
        running_loss = 0
        count = 0
        for x, y in data:
            count += 1
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            if vae:
                loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            else:
                loss = ((x - x_hat)**2).sum()
            running_loss += loss
            loss.backward()
            opt.step()
        print(f"epoch {epoch} loss: {loss}")
    return autoencoder


def test(autoencoder, data, device, thresh, vae=False):
    true, pred = [], []
    losses = []
    count = 0
    for x, y in data:
        true.append(int(y[0]))
        count += 1
        x = x.to(device) # GPU
        x_hat = autoencoder(x)

        if vae:
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
        else:
            loss = ((x - x_hat)**2).sum()
        
        if loss >= thresh:
            pred.append(1)
        else:
            pred.append(0)
        
        losses.append(loss.item())
    
    return true, pred, losses