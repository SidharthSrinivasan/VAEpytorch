import os

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt

from ssim import SSIM


# -----
# VAE Build Blocks

# #####
# encoder architecture
# #####

class Encoder(nn.Module):
    def __init__(
            self,
            latent_dim: int = 128,
            in_channels: int = 3,
    ):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        # #####
        # mu and log_var will be used as inputs for the Reparameterization Trick,
        # generating latent vector z we need
        # The encoder uses 4 CNNs all with stride 2 and padding 1
        # I found the right size for the fcs using trial and error
        # #####
        self.CNN_1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size= 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
        )
        self.CNN_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size= 3, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
        )
        self.CNN_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride = 2, padding =1),
            nn.BatchNorm2d(128),
        )
        self.CNN_4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride =2, padding = 1),
            nn.BatchNorm2d(256),
        )
        self.fc_mean = nn.Linear(1024, latent_dim)
        self.fc_log_var = nn.Linear(1024, latent_dim)


    def forward(self, x):
        x1 = F.relu(self.CNN_1(x))
        x2 = F.relu(self.CNN_2(x1))
        x3 = F.relu(self.CNN_3(x2))
        x4 = F.relu(self.CNN_4(x3))
        x5 = torch.flatten(x4, start_dim=1)
        mu = self.fc_mean(x5)
        log_var = self.fc_log_var(x5)
        return mu, log_var



class Decoder(nn.Module):
    def __init__(
            self,
            latent_dim: int = 128,
            out_channels: int = 3,
    ):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        # #####
        # I found that using transpose CNN without using output padding halves the size of the input I started with
        # #####
        self.decoder_input = nn.Linear(latent_dim, 256 * 4)
        self.Transpose_CNN1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride = 2, padding = 1, output_padding=1),
            nn.BatchNorm2d(128),
        )
        self.Transpose_CNN2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64)
        )
        self.Transpose_CNN3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32)
        )
        self.Transpose_CNN4 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32)
        )
        self.final_layer = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, padding = 1),
        )
    def forward(self, z):
        z1 = self.decoder_input(z)
        z2 = z1.view(-1, 256, 2, 2)
        z3 = self.Transpose_CNN1(z2)
        z4 = F.relu(self.Transpose_CNN2(z3))
        z5 = F.relu(self.Transpose_CNN3(z4))
        z6 = F.relu(self.Transpose_CNN4(z5))
        xg = torch.sigmoid(self.final_layer(z6))
        return xg



# #####
# Wrapper for Variational Autoencoder
# #####

class VAE(nn.Module):
    def __init__(
            self,
            latent_dim: int = 128,
    ):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encode = Encoder(latent_dim=latent_dim)
        self.decode = Decoder(latent_dim=latent_dim)

    def reparameterize(self, mu, log_var):
        """Reparameterization Tricks to sample latent vector z
        from distribution w/ mean and variance.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, x, y):
        """Forward for CVAE.
        Returns:
            xg: reconstructed image from decoder.
            mu, log_var: mean and log(std) of z ~ N(mu, sigma^2)
            z: latent vector, z = mu + sigma * eps, acquired from reparameterization trick. 
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), mu, log_var, z]



    def generate(
            self,
            n_samples: int,
    ):
        """Randomly sample from the latent space and return
        the reconstructed samples.
        Returns:
            xg: reconstructed image
            None: a placeholder simply.
        """
        rand_samples = torch.randn(n_samples, self.latent_dim)
        if torch.cuda.is_available():
          rand_samples = rand_samples.cuda()
        with torch.no_grad():
          xg = self.decode(rand_samples)
        return xg, None


# #####
# Wrapper for Conditional Variational Autoencoder
# #####

class CVAE(nn.Module):
    def __init__(
            self,
            latent_dim: int = 128,
            num_classes: int = 10,
            img_size: int = 32,
    ):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size

        # #####
        # Feel free to change parameters for encoder and decoder to suit your strategy
        # #####

        self.encode_class = nn.Linear(self.num_classes, self.img_size * self.img_size)
        self.embed_data = nn.Conv2d(3, 3, kernel_size=1)

        self.encode = Encoder(latent_dim=latent_dim, in_channels=4)
        self.decode = Decoder(latent_dim=latent_dim+10)

    def reparameterize(self, mu, log_var):
        """Reparameterization Tricks to sample latent vector z
        from distribution w/ mean and variance.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, x, y):
        # #####
        # Note that you need to process label information HERE.
        # #####
        """Forward for CVAE.
        Returns:
            xg: reconstructed image from decoder.
            mu, log_var: mean and log(std) of z ~ N(mu, sigma^2)
            z: latent vector, z = mu + sigma * eps, acquired from reparameterization trick. 
        """
        y_onehot = F.one_hot(y, 10).float()
        encoded_class = self.encode_class(y_onehot)
        embedded_class = encoded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_data = self.embed_data(x)
        x = torch.cat([embedded_data, embedded_class], dim = 1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, y_onehot], dim = 1)
        return [self.decode(z), mu, log_var, z]



    def generate(
            self,
            n_samples: int,
            y: torch.Tensor = None,
    ):
        """Randomly sample from the latent space and return
        the reconstructed samples.
        NOTE: Randomly generate some classes here, if not y is provided.
        Returns:
            xg: reconstructed image
            y: classes for xg. 
        """
        if y == None:
          y = np.random.randint(0,9,n_samples)
          y = torch.tensor(y)
        y_onehot = F.one_hot(y, 10).float()
        rand_samples = torch.randn(n_samples, self.latent_dim)
        rand_samples = torch.cat([rand_samples , y_onehot], dim= 1)
        if torch.cuda.is_available():
          rand_samples = rand_samples.cuda()
        with torch.no_grad():
          xg = self.decode(rand_samples)
        return xg, y


# #####
# Wrapper for KL Divergence
# #####

class KLDivLoss(nn.Module):
    def __init__(
            self,
            lambd: float = 1.0,
    ):
        super(KLDivLoss, self).__init__()
        self.lambd = lambd

    def forward(
            self,
            mu,
            log_var,
    ):
        loss = 0.5 * torch.sum(-log_var - 1 + mu ** 2 + log_var.exp(), dim=1)
        return self.lambd * torch.mean(loss)


# -----
# Hyperparameters
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# NOTE: Feel free to change the hyperparameters as long as you meet the marking requirement
# NOTE: DO NOT TRAIN IT LONGER THAN 100 EPOCHS.
batch_size = 128
workers = 2
latent_dim = 128
lr = 0.0001
num_epochs = 35
validate_every = 1
print_every = 100

conditional = False  # Flag to use VAE or CVAE

if conditional:
    name = "cvae"
else:
    name = "vae"

# Set up save paths
if not os.path.exists(os.path.join(os.path.curdir, "visualize", name)):
    os.makedirs(os.path.join(os.path.curdir, "visualize", name))
save_path = os.path.join(os.path.curdir, "visualize", name)
ckpt_path = name + '.pt'

kl_annealing = [0.0, 0.000001, 0.00001, 0.0001, 0.001]  # KL Annealing

# -----
# Dataset
# NOTE: Data is only normalized to [0, 1]. THIS IS IMPORTANT!!!
tfms = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=tfms)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=tfms,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers)

subset = torch.utils.data.Subset(
    test_dataset,
    [0, 380, 500, 728, 1000, 2300, 3400, 4300, 4800, 5000])

loader = torch.utils.data.DataLoader(
    subset,
    batch_size=10)

# -----
# Model
if conditional:
    model = CVAE(latent_dim=latent_dim)
else:
    model = VAE(latent_dim=latent_dim)

# -----
# Losses
l2_loss_criterion = nn.MSELoss()
bce_loss_criterion = nn.BCELoss()
ssim_loss_criterion = SSIM()
KLDiv_criterion = KLDivLoss()

# #####


best_total_loss = float("inf")

# Send to GPU
if torch.cuda.is_available():
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr)

# To further help with training
# NOTE: You can remove this if you find this unhelpful
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, [40, 50], gamma=0.1, verbose=False)


# -----
# Train loop

def train_step(x, y):
    """One train step for VAE/CVAE.
    You should return average total train loss(sum of reconstruction losses, kl divergence loss)
    and all individual average reconstruction loss (l2, bce, ssim) per sample.
    Args:
        x, y: one batch (images, labels) from Cifar10 train set.
    Returns:
        loss: total loss per batch.
        l2_loss: MSE loss for reconstruction.
        bce_loss: binary cross-entropy loss for reconstruction.
        ssim_loss: ssim loss for reconstruction.
        kldiv_loss: kl divergence loss.
    """
    optimizer.zero_grad()
    results = model(x,y)
    sample_size= len(x)
    l2_loss = l2_loss_criterion(results[0], x)
    bce_loss = bce_loss_criterion(results[0], x)
    ssim_loss = (1 - ssim_loss_criterion(results[0], x))
    kldiv_loss = KLDiv_criterion(results[1], results[2])
    recon_loss = l2_loss + bce_loss + ssim_loss
    if epoch == 1:
      KLDiv_criterion.lambd = kl_annealing[0]
    total_loss = recon_loss + (KLDiv_criterion.lambd * kldiv_loss)
    total_loss.backward()
    optimizer.step()
    
    return total_loss, l2_loss/sample_size, bce_loss/sample_size, ssim_loss/sample_size, kldiv_loss/sample_size




def denormalize(x):
    """Denomalize a normalized image back to uint8.
    Args:
        x: torch.Tensor, in [0, 1].
    Return:
        x_denormalized: denormalized image as numpy.uint8, in [0, 255].
    """

    x = x.permute(0, 2, 3, 1)
    x_denormalized = x.cpu().numpy() * 255
    x_denormalized = x_denormalized.astype(np.uint8)
    return x_denormalized


# Loop HERE
l2_losses = []
bce_losses = []
ssim_losses = []
kld_losses = []
total_losses = []

total_losses_train = []

for epoch in range(1, num_epochs + 1):
    total_loss_train = 0.0
    for i, (x, y) in enumerate(train_loader):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        # Train step
        model.train()
        loss, recon_loss, bce_loss, ssim_loss, kldiv_loss = train_step(x, y)
        total_loss_train += (loss * x.shape[0]).cpu()

        # Print
        if i % print_every == 0:
            print("Epoch {}, Iter {}: Total Loss: {:.6f} MSE: {:.6f}, SSIM: {:.6f}, BCE: {:.6f}, KLDiv: {:.6f}".format(
                epoch, i, loss, recon_loss, ssim_loss, bce_loss, kldiv_loss))

    total_losses_train.append(total_loss_train / len(train_dataset))

    # Test loop
    if epoch % validate_every == 0:
        # Loop through test set
        model.eval()

        l2_loss_per_epoch = 0
        bce_loss_per_epoch = 0
        ssim_loss_per_epoch = 0
        kld_loss_per_epoch = 0


        with torch.no_grad():
            for x, y in test_loader:
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                results = model(x,y)
                xg = results[0]
                mu = results[1]
                log_var = results[2]

                # Accumulate average reconstruction losses per batch individually for plotting

                l2_loss = l2_loss_criterion(results[0], x)
                bce_loss = bce_loss_criterion(results[0], x)
                ssim_loss = 1 - ssim_loss_criterion(results[0], x)
                kldiv_loss = KLDiv_criterion(results[1], results[2])
                avg_total_recon_loss_test= l2_loss + bce_loss + ssim_loss
                l2_loss_per_epoch+= l2_loss.cpu().numpy()
                bce_loss_per_epoch+= bce_loss.cpu().numpy()
                ssim_loss_per_epoch+= ssim_loss.cpu().numpy()
                kld_loss_per_epoch+= kldiv_loss.cpu().numpy()
            l2_losses.append(l2_loss_per_epoch/len(test_loader))
            bce_losses.append(bce_loss_per_epoch/len(test_loader))
            ssim_losses.append(ssim_loss_per_epoch/len(test_loader))
            kld_losses.append(kld_loss_per_epoch/len(test_loader))
            total_losses.append((l2_loss_per_epoch+bce_loss_per_epoch+ssim_loss_per_epoch+(KLDiv_criterion.lambd * kld_loss_per_epoch))/len(test_loader))
            
            # Plot losses
            if epoch > 1:
                plt.plot(l2_losses, label="L2 Reconstruction")
                plt.plot(bce_losses, label="BCE")
                plt.plot(ssim_losses, label="SSIM")
                plt.plot(kld_losses, label="KL Divergence")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.xlim([1, epoch])
                plt.legend()
                plt.savefig(os.path.join(os.path.join(save_path, "losses.png")), dpi=300)
                plt.clf()
                plt.close('all')

                plt.plot(total_losses, label="Total Loss Test")
                plt.plot(total_losses_train, label="Total Loss Train")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.xlim([1, epoch])
                plt.legend()
                plt.savefig(os.path.join(os.path.join(save_path, "total_loss.png")), dpi=300)
                plt.clf()
                plt.close('all')

            # Save best model
            if avg_total_recon_loss_test < best_total_loss:
                torch.save(model.state_dict(), ckpt_path)
                best_total_loss = avg_total_recon_loss_test
                print("Best model saved w/ Total Reconstruction Loss of {:.6f}.".format(best_total_loss))

        # Do some reconstruction
        model.eval()
        with torch.no_grad():
            x, y = next(iter(loader))
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # y_onehot = F.one_hot(y, 10).float()
            xg, _, _, _ = model(x, y)

            # Visualize
            xg = denormalize(xg)
            x = denormalize(x)

            y = y.cpu().numpy()

            plt.figure(figsize=(10, 5))
            for p in range(10):
                plt.subplot(4, 5, p + 1)
                plt.imshow(xg[p])
                plt.subplot(4, 5, p + 1 + 10)
                plt.imshow(x[p])
                plt.text(0, 0, "{}".format(classes[y[p].item()]), color='black',
                         backgroundcolor='white', fontsize=8)
                plt.axis('off')

            plt.savefig(os.path.join(os.path.join(save_path, "E{:d}.png".format(epoch))), dpi=300)
            plt.clf()
            plt.close('all')
            print("Figure saved at epoch {}.".format(epoch))

   
    # KL Annealing
    # Adjust scalar for KL Divergence loss
    if epoch == 1:
      KLDiv_criterion.lambd = kl_annealing[0]
    if epoch == 7:
        KLDiv_criterion.lambd = kl_annealing[1]
    elif epoch == 14:
        KLDiv_criterion.lambd = kl_annealing[2]
    elif epoch == 21:
        KLDiv_criterion.lambd = kl_annealing[3]
    elif epoch == 28:
        KLDiv_criterion.lambd = kl_annealing[4]

    print("Lambda:", KLDiv_criterion.lambd)

    # LR decay
    scheduler.step()

    print()

# Generate some random samples
if conditional:
    model = CVAE(latent_dim=latent_dim)
else:
    model = VAE(latent_dim=latent_dim)
if torch.cuda.is_available():
    model = model.cuda()
ckpt = torch.load(name + '.pt')
model.load_state_dict(ckpt)

# Generate 20 random images
xg, y = model.generate(20)
xg = denormalize(xg)
if y is not None:
    y = y.cpu().numpy()

plt.figure(figsize=(10, 5))
for p in range(20):
    plt.subplot(4, 5, p + 1)
    if y is not None:
        plt.text(0, 0, "{}".format(classes[y[p].item()]), color='black',
                 backgroundcolor='white', fontsize=8)
    plt.imshow(xg[p])
    plt.axis('off')

plt.savefig(os.path.join(os.path.join(save_path, "random.png")), dpi=300)
plt.clf()
plt.close('all')


print("Total reconstruction loss:", best_total_loss)
