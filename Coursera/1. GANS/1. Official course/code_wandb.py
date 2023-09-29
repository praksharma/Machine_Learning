import wandb
import torch

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader # to create batches
from torchvision.utils import make_grid # to create a grid of images
from torchvision import datasets,transforms as T

from torch import nn
from torchsummary import summary
# from tqdm.notebook import tqdm

wandb.login()

wandb.init(project="GANs_basic")

print("Starting the training process ...")
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 128 # Batch size during training
noise_dim = 64 # Shape of the random noise vector that we will use to generate the fake images

# Optimiser params
lr = 0.0002
beta_1 = 0.5
beta_2 = 0.99

# Training params
epochs = 5000


# Load MNIST Dataset

# various transforms to be applied on the images
train_augs = T.Compose([T.RandomRotation((-20,+20)),
                        T.ToTensor(), # (h,w,c) -> (c,h,w)
                        ])

# Download the MNIST dataset and apply the transforms
trainset = datasets.MNIST('MNIST/', download=True, train=True, transform=train_augs)


# # Checking the image and label
# image, label = trainset[5]
# plt.figure(figsize=(2,2))
# plt.imshow(image.squeeze(), cmap='gray')
# plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
# plt.savefig("sample_image.jpg", dpi = 500,bbox_inches='tight',transparent=True)
# plt.close()

print(f"Total images: {len(trainset)}")


# Load Dataset Into Batches
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True) # load the data in batches


print(f"Total no. of batches: {len(trainloader)}")


dataiter = iter(trainloader)
images,_ = next(dataiter) # get the next batch of images

print(images.shape) # number of images in a batch, no. of channels, height, width


# 'show_tensor_images' : function is used to plot some of images from the batch

def show_tensor_images(tensor_img, savename, num_images = 16, size=(1, 28, 28)):
    unflat_img = tensor_img.detach().cpu()
    img_grid = make_grid(unflat_img[:num_images], nrow=4)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze())
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.savefig("plots/"+savename, dpi = 500,bbox_inches='tight',transparent=True)   
    plt.close()


show_tensor_images(images,"sample_images.jpg")


# # Create Discriminator Network

def get_disc_block(in_channels, out_channels, kernel_size, stride):
    """Discriminator block"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2))


# Now we can create a discriminator network for the training images
class Discriminator(nn.Module):
    """Discriminator Network"""
    def __init__(self):
        super().__init__()

        # Declaring variables we need for the forward pass
        self.block_1 = get_disc_block(in_channels = 1, out_channels = 16, kernel_size = (3,3), stride = 2)
        self.block_2 = get_disc_block(in_channels = 16, out_channels = 32, kernel_size = (5,5), stride = 2)
        self.block_3 = get_disc_block(in_channels = 32, out_channels = 64, kernel_size = (5,5), stride = 2)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features = 64, out_features = 1)

    def forward(self, images):
        """Forward pass"""

        # Convolution layers
        x1 = self.block_1(images)
        x2 = self.block_2(x1)
        x3 = self.block_3(x2)

        # Flatten and pass it through the linear layer
        x4 = self.flatten(x3)
        x5 = self.linear(x4)
        
        return x5


D = Discriminator().to(device)
summary(D, input_size=(1, 28, 28))


# Create Generator Network

def get_gen_block(in_channels, out_channels, kernel_size, stride, final_block = False):
    """Generator block
    All layers are same except the last layer which uses tanh activation function and no batch norm, so we will use an if condition
    """
    if final_block:
        return nn.Sequential(
        nn.ConvTranspose2d(in_channels , out_channels , kernel_size , stride),
        nn.Tanh(),
    )
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels , out_channels , kernel_size , stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

# Class to generate the fake images
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super().__init__()

        self.noise_dim = noise_dim
        # Declaring variables we need for the forward pass
        self.block_1 = get_gen_block(in_channels = noise_dim, out_channels = 256, kernel_size = (3,3), stride = 2)
        self.block_2 = get_gen_block(in_channels = 256, out_channels = 128, kernel_size = (4,4), stride = 1)
        self.block_3 = get_gen_block(in_channels = 128, out_channels = 64, kernel_size = (3,3), stride = 2)
        self.block_4 = get_gen_block(in_channels = 64, out_channels = 1, kernel_size = (4,4), stride = 2, final_block = True)

    def forward(self, random_noise):
        """forward pass"""
        # random_noise -> (bs,noise_dim) -> (bs, noise_dim , 1 , 1)
        x = random_noise.view(-1, self.noise_dim, 1, 1) 
        # Convolutional layers
        
        x1 = self.block_1(x)
        x2 = self.block_2(x1)
        x3 = self.block_3(x2)
        x4 = self.block_4(x3)

        return x4


G = Generator(noise_dim).to(device)
summary(G, input_size=(1, noise_dim))

# Replace Random initialized weights to Normal weights

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02) # mean = 0.0, std = 0.02
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02) # mean = 0.0, std = 0.02
        nn.init.constant_(m.bias, 0)

D = D.apply(weights_init)
G = G.apply(weights_init)


# # Create Loss Function and Load Optimizer
# * real loss function
# * fake loss function


def real_loss(disc_pred):
    criterion = nn.BCEWithLogitsLoss()
    ground_truth = torch.ones_like(disc_pred) # all ones
    loss = criterion(disc_pred, ground_truth)
    return loss

def fake_loss(disc_pred):
    criterion = nn.BCEWithLogitsLoss()
    ground_truth = torch.zeros_like(disc_pred) # all zeros
    loss = criterion(disc_pred, ground_truth)
    return loss


# discriminator and generator optimizers
D_opt = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta_1, beta_2)) #defined in the first cell
G_opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta_1, beta_2)) #defined in the first cell


# # Training Loop
# Now we have all the recipes to train the neural network system


total_d_loss_array = data_loss_array = np.zeros(epochs)
total_g_loss_array = data_loss_array = np.zeros(epochs)

for i in range(epochs):
    # initialize loss variables before each batch
    total_d_loss = 0.0
    total_g_loss = 0.0

    # batch training
    for real_img, _ in (trainloader):
        real_img = real_img.to(device) # move the images to the CUDA device
        noise = torch.randn(batch_size, noise_dim).to(device) # generate random noise for the generator

        ### Train the discriminator
        D_opt.zero_grad() # avoid gradient accumulation

        # see the first image of the notebook
        fake_img = G(noise) # generate fake images from the noise
        D_pred = D(fake_img) # pass the fake images to the discriminator

        D_fake_loss = fake_loss(D_pred) # pass the fake images to the discriminator

        D_pred = D(real_img)
        D_real_loss = real_loss(D_pred) # pass the real images to the discriminator

        D_loss = (D_fake_loss + D_real_loss)/2 # average loss
        total_d_loss += D_loss.item() # accumulate the discriminator loss

        D_loss.backward() # gradients for discriminator

        D_opt.step() # update discriminator weights

        ### Train the generator
        G_opt.zero_grad() # avoid gradient accumulation
        noise = torch.randn(batch_size, noise_dim).to(device) # generate random noise for the generator

        fake_img = G(noise) # generate fake images from the noise
        D_pred = D(fake_img) # pass the fake images to the discriminator

        G_loss = real_loss(D_pred) # we want the generator to create real images
        total_g_loss += G_loss.item() # accumulate the generator loss

        G_loss.backward() # gradients for generator
        G_opt.step() # update generator weights
    
    avg_d_loss = total_d_loss/len(trainloader) # average discriminator loss over all batches
    avg_g_loss = total_g_loss/len(trainloader) # average generator loss over all batches

    total_d_loss_array[i] = total_d_loss
    total_g_loss_array[i] = total_g_loss

    print(f"Epoch: {i+1}/{epochs} | D loss: {avg_d_loss:.4f} | G loss: {avg_g_loss:.4f}")
    #show_tensor_images(fake_img)

    wandb.log({'epoch': i + 1,
        'D loss': avg_d_loss,
        'G loss': avg_g_loss,})
        


plt.figure(dpi=150)
indices = np.arange(0,epochs,1) 
iter_array = np.arange(0,epochs)

plt.plot(iter_array[indices],total_d_loss_array[indices], linewidth=1)
plt.plot(iter_array[indices],total_g_loss_array[indices], linewidth=1)
plt.legend(['D Loss', 'G Loss'])
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.savefig("plots/"+'Loss_curve.jpg', dpi = 500,bbox_inches='tight',transparent=True)
plt.close()

# Run after training is completed.
# Now you can use Generator Network to generate handwritten images

noise = torch.randn(batch_size, noise_dim, device = device)
generated_image = G(noise)

show_tensor_images(generated_image, "final_generated_image.jpg")

# Mark the run as finished (useful in Jupyter notebooks)
wandb.finish()


