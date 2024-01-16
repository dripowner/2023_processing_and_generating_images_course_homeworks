import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import tqdm


class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(3*784, 512)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 3*784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 3, 28, 28))
    

class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(3*784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
 

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))
                
        return x


class ConvVAE(nn.Module):
    def __init__(self,
                 latent_dim=16):
        super(ConvVAE, self).__init__()

        self.criterion = nn.MSELoss()

        self.Encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=5, 
                stride=1, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=5, 
                stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, 
                stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=5, 
                stride=2, padding=0
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        # fully connected layers for learning representations
        self.fc1 = nn.Linear(256, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_log_var = nn.Linear(32, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 32)
        self.fc3 = nn.Linear(32, 256)

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=5, 
                stride=1, padding=0
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=32, kernel_size=5, 
                stride=2, padding=0
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=3, kernel_size=4, 
                stride=2, padding=0
            ),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
            nn.Dropout(0.25)
        )

    def encoder(self, x):
        return self.Encoder(x)
    
    def decoder(self, x):
        return self.Decoder(x)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
    
    def loss(self, reconstruction, mu, logvar, x):
        bce_loss = self.criterion(reconstruction, x)
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss += bce_loss
        return loss
 
    def forward(self, x):
        # encoding
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, 1)
        hidden = self.fc1(encoded)

        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        z = self.fc3(z)
        z = z.view(-1, 256, 1, 1)
 
        # decoding
        reconstruction = self.decoder(z)

        return reconstruction, mu, log_var
    

class MyModel(nn.Module):
    def __init__(self, latent_dim=32, dropout=0.15) -> None:
        super().__init__()

        self.criterion = nn.MSELoss()

        self.Encoder = nn.Sequential(
            nn.Conv2d(3, 6, 3, stride=2, padding=1), # 14x14
            nn.ReLU(),
            # nn.Conv2d(8, 8, 3, stride=2, padding=1), # 7x7
            # nn.ReLU(),
            # nn.MaxPool2d(3, stride=2, padding=1), # 7x7
        )
        self.fc1 = nn.Linear(6*14*14, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_log_var = nn.Linear(512, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 6*14*14)

        self.Decoder = nn.Sequential(
            nn.Linear(6*14*14, 3*28*28),
            nn.Sigmoid(),
        )

    def encoder(self, x):
        return self.Encoder(x)
    
    def decoder(self, x):
        return self.Decoder(x)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
    
    def loss(self, reconstruction, x):
        return self.criterion(reconstruction, x)
 
    def forward(self, x):
        # encoding
        encoded = self.encoder(x)
        hidden = torch.flatten(encoded, 1)

        hidden = self.fc1(hidden)
        hidden = F.relu(hidden)

        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        z = self.fc2(z)
        z = F.relu(z)

        z = z.view(-1, 6, 14, 14)

        z = torch.flatten(z, 1)
 
        # decoding
        reconstruction = self.decoder(z)
        reconstruction = reconstruction.view((-1, 3, 28, 28))

        return reconstruction, mu, log_var
