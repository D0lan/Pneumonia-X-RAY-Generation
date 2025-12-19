#!/usr/bin/env python
# coding: utf-8

# -------- VAE for Chest X-Ray (NORMAL + PNEUMONIA, SEPARATE OUTPUTS) ----------

import os
from pathlib import Path
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# -----------------------
# Parameters
# -----------------------
train_dir = "/home/csgrad/nancyipo/Desktop/VAE_676/chest_xray/train"
test_dir  = "/home/csgrad/nancyipo/Desktop/VAE_676/chest_xray/test"

output_dir = "output3"
recon_dir = "output3/reconstructions3"
gen_dir = "output3/generated3"
ckpt_dir = "output3/checkpoints3"

img_size = 128
channels = 1
latent_dim = 128
batch_size = 32
epochs = 200
lr = 1e-3
recon_weight = 1.0
save_every = 5
num_workers = 2
force_cpu = False


# -----------------------
# Dataset (WITH LABELS)
# -----------------------
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None, exts=(".jpg", ".jpeg", ".png")):
        self.root_dir = Path(root_dir)

        ### NEW: Collect NORMAL and PNEUMONIA separately
        self.files = []
        for label_name, label in [("NORMAL", 0), ("PNEUMONIA", 1)]:
            class_dir = self.root_dir / label_name
            for p in class_dir.rglob("*"):
                if p.suffix.lower() in exts:
                    self.files.append((p, label))

        if len(self.files) == 0:
            raise RuntimeError(f"No images in {root_dir}")

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label   # <--- IMPORTANT


# -----------------------
# VAE Model
# -----------------------
class ConvVAE(nn.Module):
    def __init__(self, img_channels=1, hidden_dims=None, latent_dim=128):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        modules = []
        in_channels = img_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, 4, 2, 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1]*8*8, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1]*8*8, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*8*8)

        hidden_dims_rev = list(reversed(hidden_dims))
        modules = []
        for i in range(len(hidden_dims_rev)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims_rev[i], hidden_dims_rev[i+1], 4, 2, 1),
                    nn.BatchNorm2d(hidden_dims_rev[i+1]),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims_rev[-1], img_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 256, 8, 8)
        h = self.decoder(h)
        return self.final_layer(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# -----------------------
# Loss
# -----------------------
def loss_function(recon_x, x, mu, logvar, recon_weight=1.0):
    bce = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_weight*bce + kld, bce, kld


# -----------------------
# Utils
# -----------------------
def make_output_dirs():
    os.makedirs(recon_dir + "/normal", exist_ok=True)
    os.makedirs(recon_dir + "/pneumonia", exist_ok=True)
    os.makedirs(gen_dir + "/normal", exist_ok=True)
    os.makedirs(gen_dir + "/pneumonia", exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)


# -----------------------
# Training + Testing
# -----------------------
def train_and_test():
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # Load datasets
    train_dataset = ImageFolderDataset(train_dir, transform)
    test_dataset = ImageFolderDataset(test_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ConvVAE(img_channels=channels, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    make_output_dirs()

    # -----------------------
    # Training Loop
    # -----------------------
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(imgs)
            loss, _, _ = loss_function(recon, imgs, mu, logvar)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss/len(train_loader.dataset):.4f}")

        # Save reconstruction examples
        if epoch % save_every == 0:
            model.eval()
            with torch.no_grad():
                imgs, labels = next(iter(train_loader))
                imgs = imgs.to(device)
                recon, _, _ = model(imgs)

                for i in range(min(8, imgs.size(0))):
                    label = labels[i].item()
                    folder = "normal" if label == 0 else "pneumonia"
                    utils.save_image(
                        torch.cat([imgs[i], recon[i]], dim=2).cpu(),
                        f"{recon_dir}/{folder}/epoch_{epoch}_idx_{i}_label_{label}.png"
                    )

            torch.save(model.state_dict(), f"{ckpt_dir}/vae_epoch_{epoch}.pt")


    # -----------------------
    # TEST Reconstruction
    # -----------------------
    print("\nRunning test...")
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            recon, _, _ = model(imgs)

            for i in range(min(8, imgs.size(0))):
                label = labels[i].item()
                folder = "normal" if label == 0 else "pneumonia"
                utils.save_image(
                    torch.cat([imgs[i], recon[i]], dim=2).cpu(),
                    f"{recon_dir}/{folder}/test_idx_{i}_label_{label}.png"
                )
            break


    # -----------------------
    # GENERATE IMAGES
    # -----------------------
    print("Generating samples...")

    with torch.no_grad():
        # NORMAL (label 0)
        z = torch.randn(16, latent_dim).to(device)
        samples = model.decode(z)
        utils.save_image(samples.cpu(), f"{gen_dir}/normal/generated_normal.png")

        # PNEUMONIA (label 1)
        z = torch.randn(16, latent_dim).to(device)
        samples = model.decode(z)
        utils.save_image(samples.cpu(), f"{gen_dir}/pneumonia/generated_pneumonia.png")

    print("\nDONE! Outputs saved in:", output_dir)



# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    train_and_test()
