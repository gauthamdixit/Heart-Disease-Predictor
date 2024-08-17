import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset from the file
data = np.loadtxt('dataset_', delimiter=',')

# Separate features and target
X = data[:, :-1]
y = data[:, -1]

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Split the dataset into training (70%), validation (15%), and test (15%) sets
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
test_size = len(X) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(TensorDataset(X, y), [train_size, val_size, test_size])

# Create DataLoaders for training, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(X.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.silu(self.fc1(x))
        x = self.silu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x

par_dict = {}
model = NN()
criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 300
epoch_list = []
train_losses = []
val_losses = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    train_loss = running_loss/len(train_loader)
    validation_loss = val_loss/len(train_loader)
    train_losses.append(train_loss)
    val_losses.append(validation_loss)
    epoch_list.append(epoch)

    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss}, Validation Loss: {validation_loss}")

print("Training complete.")

plt.figure(figsize=(10, 6))
plt.plot(epoch_list, train_losses, label='Training Loss', color='blue')
plt.plot(epoch_list, val_losses, label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig("losses.png")
plt.show()


# Test the model
model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        predictions.extend(outputs.numpy().flatten())
        actuals.extend(targets.numpy().flatten())

predictions = np.array(predictions)
actuals = np.array(actuals)



rounded_preds = np.abs(np.round(predictions))
for pred, actual in zip(rounded_preds, actuals):
    print(f"Prediction: {pred:.2f}, Target: {actual:.2f}")
accuracy = accuracy_score(actuals, rounded_preds)

print(f"Test Set Accuracy: {accuracy:.2f}")
# Save the model's state dictionary
torch.save(model.state_dict(), 'model.pth')

print("Model saved successfully.")
