import torch
import torch.nn as nn
import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
import os
import matplotlib.pyplot as plt

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "baseline_results/mnist_cnn_baseline.pth"
output_dir = "attack_data"
os.makedirs(output_dir, exist_ok=True)
model = MNIST_CNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE).eval()
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
x_test, y_test = next(iter(test_loader))
x_test_np = x_test.numpy()
y_test_np = y_test.numpy()

classifier = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    input_shape=(1, 28, 28),
    nb_classes=10,
    clip_values=(0, 1)  
)

# generate adversarial samples using FGSM  
# epsilon is magnitude of the perturbation
epsilon = 0.2
attack = FastGradientMethod(estimator=classifier, eps=epsilon, targeted=False)
x_test_adversarial = attack.generate(x=x_test_np)
np.savez(os.path.join(output_dir, 'adversarial_test_set.npz'), x=x_test_adversarial, y=y_test_np)
preds_clean = np.argmax(classifier.predict(x_test_np), axis=1)
preds_adversarial = np.argmax(classifier.predict(x_test_adversarial), axis=1)
successful_attacks = np.where(preds_clean != preds_adversarial)[0]

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle(f"FGSM Attack Examples (Epsilon={epsilon})")
for i, ax_row in enumerate(axes):
    for j, ax in enumerate(ax_row):
        idx = successful_attacks[i*5 + j]
        if i == 0:
            ax.imshow(x_test_np[idx].squeeze(), cmap='gray')
            ax.set_title(f"Clean (Pred: {preds_clean[idx]})")
        else:
            ax.imshow(x_test_adversarial[idx].squeeze(), cmap='gray')
            ax.set_title(f"Adversarial (Pred: {preds_adversarial[idx]})")
        ax.axis('off')
plt.savefig(os.path.join(output_dir, "fgsm_attack_visualization.png"))
plt.show()
