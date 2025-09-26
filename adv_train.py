import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7*7*64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 7*7*64)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def evaluate(model, device, test_loader, description="Test Set", needs_normalization=False):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            if needs_normalization:
                data = (data - 0.1307) / 0.3081
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'{description}: Avg Loss: {avg_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy, avg_loss, all_targets, all_preds

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "adversarial_training_results"
os.makedirs(output_dir, exist_ok=True)
EPSILON = 0.25
transform_normalized = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
transform_unnormalized = transforms.ToTensor()
train_dataset_unnormalized = datasets.MNIST(root='./data', train=True, download=True, transform=transform_unnormalized)
train_dataset_normalized = datasets.MNIST(root='./data', train=True, download=True, transform=transform_normalized)
test_dataset_normalized = datasets.MNIST(root='./data', train=False, download=True, transform=transform_normalized)
x_train_np = train_dataset_unnormalized.data.numpy() / 255.0
x_train_np = np.expand_dims(x_train_np, 1).astype(np.float32)
y_train_np = train_dataset_unnormalized.targets.numpy()

temp_model = MNIST_CNN().to(DEVICE)
optimizer = optim.Adam(temp_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
temp_loader = DataLoader(train_dataset_normalized, batch_size=128, shuffle=True)

for _ in range(5):  
    temp_model.train()
    for data, target in temp_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = temp_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

classifier = PyTorchClassifier(
    model=temp_model, loss=criterion, optimizer=optimizer,
    input_shape=(1, 28, 28), nb_classes=10, clip_values=(0, 1)
)
attack = FastGradientMethod(estimator=classifier, eps=EPSILON, targeted=False)
print("Generating adversarial training samples")
x_train_adv = attack.generate(x=x_train_np)

x_adv_tensor = torch.from_numpy(x_train_adv)
y_train_tensor = torch.from_numpy(y_train_np).long()
adversarial_train_dataset = TensorDataset(x_adv_tensor, y_train_tensor)
clean_train_dataset = TensorDataset(torch.from_numpy(x_train_np), y_train_tensor)
combined_dataset = ConcatDataset([clean_train_dataset, adversarial_train_dataset])
combined_loader = DataLoader(combined_dataset, batch_size=128, shuffle=True)

robust_model = MNIST_CNN().to(DEVICE)
optimizer = optim.Adam(robust_model.parameters(), lr=0.001)
for epoch in range(1, 6):
    robust_model.train()
    for batch_idx, (data, target) in enumerate(combined_loader):
        data = (data - 0.1307) / 0.3081
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = robust_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Training Epoch: {epoch}, Final Batch Loss: {loss.item():.6f}")

torch.save(robust_model.state_dict(), os.path.join(output_dir, 'mnist_cnn_robust.pth'))

clean_test_loader = DataLoader(test_dataset_normalized, batch_size=1000, shuffle=False)
acc_clean, loss_clean, targets_clean, preds_clean = evaluate(robust_model, DEVICE, clean_test_loader, description="Clean Test Set")
adv_test_data = np.load("attack_data/adversarial_test_set.npz")
adv_loader = DataLoader(TensorDataset(torch.from_numpy(adv_test_data['x']).float(), torch.from_numpy(adv_test_data['y']).long()), batch_size=1000)
acc_adv, loss_adv, targets_adv, preds_adv = evaluate(robust_model, DEVICE, adv_loader, description="Adversarial Test Set", needs_normalization=True)
cm_clean = confusion_matrix(targets_clean, preds_clean)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_clean, annot=True, fmt='d', cmap='Greens', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Robust Model Confusion Matrix on CLEAN Data')
plt.savefig(os.path.join(output_dir, 'confusion_matrix_robust_clean.png'))
plt.show()

cm_adv = confusion_matrix(targets_adv, preds_adv)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_adv, annot=True, fmt='d', cmap='Reds', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Robust Model Confusion Matrix on ADVERSARIAL Data')
plt.savefig(os.path.join(output_dir, 'confusion_matrix_robust_adversarial.png'))
plt.show()

with open(os.path.join(output_dir, 'robust_model_performance.txt'), 'w') as f:
    f.write('Robust Model Performance After Adversarial Training\n')
    f.write('='*50 + '\n')
    f.write(f'Accuracy on Clean Test Set: {acc_clean:.2f}%\n')
    f.write(f'Average Loss on Clean Test Set: {loss_clean:.4f}\n\n')
    f.write(f'Accuracy on Adversarial Test Set: {acc_adv:.2f}%\n')
    f.write(f'Average Loss on Adversarial Test Set: {loss_adv:.4f}\n\n')
