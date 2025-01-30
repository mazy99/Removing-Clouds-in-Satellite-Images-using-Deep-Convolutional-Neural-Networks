import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn.functional as F


# 1. Определение Improved Tversky Loss
def tversky_loss_multiclass(logits, target, alpha=0.5, beta=0.5, smooth=1e-6):
    """
    Tversky loss for multi-classification tasks with dynamic weights based on class distribution.

    Args:
        logits (torch.Tensor): The output from the model (batch_size, num_classes, H, W)
        target (torch.Tensor): Ground truth masks (batch_size, H, W)
        alpha (float): Weight for false positives
        beta (float): Weight for false negatives
        smooth (float): Smoothing factor to avoid division by zero
    Returns:
        torch.Tensor: The computed loss
    """
    num_classes = logits.shape[1]

    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    # Flatten tensors
    logits = torch.clamp(logits, 1e-6, 1 - 1e-6).view(logits.size(0), num_classes, -1)
    target_one_hot = target_one_hot.view(target_one_hot.size(0), num_classes, -1)

    # Calculate TP, FP, and FN
    TP = (logits * target_one_hot).sum(dim=-1)
    FP = (logits * (1 - target_one_hot)).sum(dim=-1)
    FN = ((1 - logits) * target_one_hot).sum(dim=-1)

    # Calculate class-wise Tversky
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    # Calculate dynamic class weights
    class_weights = (1.0 / (target_one_hot.sum(dim=(0, 2)) + smooth))
    class_weights /= class_weights.sum()
    class_weights = class_weights.detach()
    # Weighted average loss
    loss = (1 - tversky) * class_weights
    loss = loss.mean()

    return loss


def cross_entropy_loss(logits, target):
    """
    Cross entropy loss for multi-classification tasks.

    Args:
        logits (torch.Tensor): The output from the model (batch_size, num_classes, H, W)
        target (torch.Tensor): Ground truth masks (batch_size, H, W)

    Returns:
        torch.Tensor: The computed loss
    """
    loss = F.cross_entropy(logits, target)
    return loss


def t_net_loss(logits, target, alpha=0.5, beta=0.5, smooth=1e-6):
    """
    Combined loss function for T-Net

    Args:
        logits (torch.Tensor): The output from the model (batch_size, num_classes, H, W)
        target (torch.Tensor): Ground truth masks (batch_size, H, W)
        alpha (float): Weight for false positives
        beta (float): Weight for false negatives
        smooth (float): Smoothing factor to avoid division by zero
        Returns:
        torch.Tensor: The computed loss
    """
    cross_entropy = cross_entropy_loss(logits, target)
    tversky = tversky_loss_multiclass(logits, target, alpha, beta, smooth)

    loss = 0.5 * (cross_entropy + tversky)
    return loss


# 2. Определение архитектуры UNet (T-Net) c ASPP
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        # Global Average Pooling
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        self.convs = nn.ModuleList(modules)

        # 1x1 convolution for fusion
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res[4] = F.interpolate(res[4], size=x.size()[2:], mode='bilinear',
                               align_corners=True)  # upsample global avg pooling
        res = torch.cat(res, dim=1)
        return self.project(res)


class DownWithASPP(nn.Module):
    """Downscaling with maxpool then double conv then ASPP and residual connection"""

    def __init__(self, in_channels, out_channels, atrous_rates):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        self.aspp = ASPP(out_channels, out_channels, atrous_rates)
        self.residual_connection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x_down = self.maxpool_conv(x)
        x_aspp = self.aspp(x_down)
        x_res = self.residual_connection(x)
        x_res = F.interpolate(x_res, size=x_aspp.size()[2:], mode='bilinear', align_corners=True)
        print(f"x shape: {x.shape}, x_down shape: {x_down.shape}")
        print(f"x_aspp shape: {x_aspp.shape}, x_res shape: {x_res.shape}")
        return x_aspp + x_res


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        atrous_rates = [6, 12, 18]

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = DownWithASPP(128, 256, atrous_rates)
        self.down3 = DownWithASPP(256, 512, atrous_rates)
        factor = 2 if bilinear else 1
        self.down4 = DownWithASPP(512, 1024 // factor, atrous_rates)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


# 3. Создание кастомного Dataset
class CloudSegmentationDataset(Dataset):
    def __init__(self, image_dir, cloud_mask_dir, shadow_mask_dir, transform=None):
        self.image_dir = image_dir
        self.cloud_mask_dir = cloud_mask_dir
        self.shadow_mask_dir = shadow_mask_dir
        self.image_files = []

        # Проверка наличия масок для каждого изображения
        for img_name in sorted(os.listdir(image_dir)):
            if img_name.endswith('.tiff'):
                cloud_mask_name = img_name.replace('.tiff', '-c.npy')
                cloud_mask_path = os.path.join(cloud_mask_dir, cloud_mask_name)

                shadow_mask_name = img_name.replace('.tiff', '-s.npy')
                shadow_mask_path = os.path.join(shadow_mask_dir, shadow_mask_name)

                if os.path.exists(cloud_mask_path) and os.path.exists(shadow_mask_path):
                    self.image_files.append(img_name)

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        cloud_mask_name = img_name.replace('.tiff', '-c.npy')
        cloud_mask_path = os.path.join(self.cloud_mask_dir, cloud_mask_name)

        shadow_mask_name = img_name.replace('.tiff', '-s.npy')
        shadow_mask_path = os.path.join(self.shadow_mask_dir, shadow_mask_name)

        image = io.imread(img_path)  # [H, W, C]
        cloud_mask = np.load(cloud_mask_path)  # [H, W]
        shadow_mask = np.load(shadow_mask_path)  # [H, W]

        # Объединяем маски в один канал, где:
        # 0 - нет облаков и теней
        # 1 - облака
        # 2 - тени
        mask = np.zeros_like(cloud_mask, dtype=np.int64)
        mask[cloud_mask > 0] = 1
        mask[shadow_mask > 0] = 2

        # Применяем трансформации если есть
        if self.transform:
            image = self.transform(image)

            mask = torch.from_numpy(mask).long()  # преобразуем в long tensor

        return image, mask


# 4. Функция обучения
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
    model.to(device)
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                epoch_val_loss += loss.item()
        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

    return train_losses, val_losses


# 5. Визуализация потерь
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# 6. Основной блок
if __name__ == '__main__':
    # Параметры
    image_dir = 'original_igms'  # Замените на путь к вашим tiff изображениям
    cloud_mask_dir = 'cloud_mask'  # Замените на путь к маскам облаков (.npy)
    shadow_mask_dir = 'shadow_mask'  # Замените на путь к маскам теней (.npy)

    batch_size = 4
    num_epochs = 20
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_channels = 15  # количество каналов изображения
    n_classes = 3  # количество классов 0 - нет облаков/теней, 1 - облака, 2 - тени

    # Трансформации для изображений
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float)
        # Можно добавить нормализацию, если требуется
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Загрузка данных
    dataset = CloudSegmentationDataset(image_dir, cloud_mask_dir, shadow_mask_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Инициализация модели, оптимизатора и функции потерь
    model = UNet(n_channels, n_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = t_net_loss

    # Обучение модели
    train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device)
    plot_loss(train_losses, val_losses)

    # Сохранение обученной модели
    torch.save(model.state_dict(), 'cloud_segmentation_model.pth')