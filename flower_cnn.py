import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torchvision.models as models
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define hyperparameters
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
NUM_CLASSES = 5
VALID_SPLIT = 0.2  # 20% of the training data will be used for validation
NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data transformations
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the CNN model
class FlowerCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(FlowerCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fourth convolutional block
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fifth convolutional block
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after all the convolution and pooling layers
        # After 5 pooling layers with stride 2, the size is reduced by a factor of 2^5 = 32
        fc_input_size = 512 * (IMAGE_SIZE // 32) * (IMAGE_SIZE // 32)
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.relu6 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Store intermediate activations for feature visualization
        self.feature_maps = {}
        
        # First block
        x = self.conv1(x)
        self.feature_maps['conv1'] = x.detach()
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Second block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Third block
        x = self.conv3(x)
        self.feature_maps['conv3'] = x.detach()
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Fourth block
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        # Fifth block
        x = self.conv5(x)
        self.feature_maps['conv5'] = x.detach()
        x = self.relu5(x)
        x = self.pool5(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# Evaluation function
def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    return epoch_loss, accuracy, precision, recall, f1

# Function to visualize feature maps
def visualize_features(model, data_loader, layer_name, num_features=8):
    # Get a batch of images
    inputs, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    
    # Forward pass to get feature maps
    _ = model(inputs)
    
    # Get the feature maps for the specified layer
    feature_maps = model.feature_maps[layer_name]
    
    # Move feature maps to CPU and convert to numpy
    feature_maps = feature_maps.cpu().numpy()
    
    # Get the first image's feature maps
    image_features = feature_maps[0]
    
    # Create a figure to display the feature maps
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    # Display a subset of feature maps
    for i in range(min(num_features, image_features.shape[0])):
        feature_map = image_features[i]
        axes[i].imshow(feature_map, cmap='viridis')
        axes[i].set_title(f"Filter {i+1}")
        axes[i].axis('off')
    
    plt.suptitle(f"Feature Maps from {layer_name}")
    plt.tight_layout()
    plt.savefig(f"{layer_name}_features.png")
    plt.close()

def main():
    # Load the dataset
    print("Loading dataset...")
    dataset = datasets.ImageFolder(root='train', transform=train_transform)

    # Get class names
    class_names = dataset.classes
    print(f"Classes: {class_names}")

    # Split the dataset into training and validation sets
    total_size = len(dataset)
    val_size = int(total_size * VALID_SPLIT)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create validation dataset with test transform
    val_dataset.dataset.transform = test_transform

    print(f"Total dataset size: {total_size}")
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize the model
    model = FlowerCNN(num_classes=NUM_CLASSES).to(device)
    print(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("Starting training...")
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Record start time
    start_time = time.time()

    for epoch in range(EPOCHS):
        # Train
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate
        val_loss, val_acc, precision, recall, f1 = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    # Calculate and save training time
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Save training time to file
    with open('flower_cnn_train_time.txt', 'w') as f:
        f.write(f"{train_time:.2f}")

    # Save the model
    torch.save(model.state_dict(), 'flower_cnn_model.pth')
    print("Model saved as flower_cnn_model.pth")

    # Visualize training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

    # Visualize feature maps from different layers
    print("Visualizing feature maps...")
    visualize_features(model, val_loader, 'conv1')
    visualize_features(model, val_loader, 'conv3')
    visualize_features(model, val_loader, 'conv5')

    # Final evaluation
    final_loss, final_acc, final_precision, final_recall, final_f1 = evaluate(model, val_loader, criterion, device)
    print("\nFinal Model Performance:")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall: {final_recall:.4f}")
    print(f"F1 Score: {final_f1:.4f}")

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

if __name__ == "__main__":
    main() 