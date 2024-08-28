import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class FoInternNet(nn.Module):
    def __init__(self, input_size, n_classes, dropout_prob=0.5):
        super(FoInternNet, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes

        # Havuzlamalı evrişimsel katmanlar
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Dropout katmanı (overfitting’i önlemek için)
        self.dropout = nn.Dropout(p=dropout_prob)

        # Transpoze edilmiş evrişimsel katmanlar (Yukarı Örnekleme)
        self.upconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=n_classes, kernel_size=2, stride=2)

    def forward(self, x):
        """Giriş verilerini tanımlanan model katmanlarına aktarır."""
        # Alt örnekleme yolu
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Dropout uygulaması
        x = self.dropout(x)

        # Örnekleme yolu
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = self.upconv3(x)

        return x

input_size = 224
# Veri Artırma (Data Augmentation) Teknikleri
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Görüntüleri yatay olarak rastgele çevirir
    transforms.RandomRotation(10),      # Görüntüleri rastgele 10 derece döndürür
    transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),  # Görüntüleri rastgele kırpıp yeniden boyutlandırır
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Parlaklık ve kontrastı rastgele değiştirir
    transforms.ToTensor()  # Görüntüleri tensöre dönüştürür
])