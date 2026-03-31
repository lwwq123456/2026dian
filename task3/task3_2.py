import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from task3_1 import GatedDeltaRuleRecurrent

class GDNBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)

        self.gdn = GatedDeltaRuleRecurrent(d_model) 
        self.drop1 = nn.Dropout(0.1)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = x + self.drop1(self.gdn(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x

class GDNVision(nn.Module):
    def __init__(self, d_model=128, num_classes=10, num_layers=4):
        super().__init__()
        self.patch_embed = nn.Linear(28, d_model)

        self.blocks = nn.ModuleList([GDNBlock(d_model) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.classifier(x[:, -1, :])


def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动任务 3.2 (Serial Baseline) | 设备: {device}")
    
    model = GDNVision(d_model=128, num_layers=4).to(device)
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=transform), batch_size=512, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=transform), batch_size=512, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    epochs = 15
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_hist, acc_hist = [], []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        loss_hist.append(avg_loss)
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        acc_hist.append(acc)
        print(f"Epoch [{epoch+1:>2d}/{epochs}] | Loss: {avg_loss:.4f} | Accuracy: {acc*100:>5.2f}%")

    print(f"🎉 训练完成！总耗时: {time.time() - start_time:.1f} 秒 | 最高准确率: {max(acc_hist)*100:.2f}%")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=120)
    ax1.plot(range(1, epochs+1), loss_hist, 'b-o', label='Train Loss')
    ax1.set_xlabel('Epochs', fontweight='bold')
    ax1.set_ylabel('Loss', color='b', fontweight='bold')
    ax2 = ax1.twinx()
    ax2.plot(range(1, epochs+1), acc_hist, 'r-s', label='Test Accuracy')
    ax2.set_ylabel('Accuracy', color='r', fontweight='bold')
    plt.title('Task 3.2: Serial Baseline Metrics', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig('task3_2_baseline.png')

if __name__ == "__main__":
    train_and_evaluate()