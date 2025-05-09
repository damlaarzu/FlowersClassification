import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import cv2
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001  # Lower learning rate for fine-tuning
IMAGE_SIZE = 224  # VGG16 expects 224x224 images
NUM_CLASSES = 5
VALID_SPLIT = 0.2  # 20% of the training data will be used for validation
NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data transformations (using VGG16 normalization values)
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

# Inverse normalization for visualization
inverse_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

# Define the VGG16 Fine-Tuning model
class VGG16FineTuning(nn.Module):
    def __init__(self, num_classes=5):
        super(VGG16FineTuning, self).__init__()
        
        # Load pretrained VGG16 model
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # Store feature maps for visualization
        self.feature_maps = {}
        
        # Replace the final classifier layer for our specific number of classes
        num_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_features, num_classes)
        
        # Freeze only the first convolutional block (first 4 layers)
        # VGG16 features structure: conv1_1, relu1_1, conv1_2, relu1_2, maxpool1, ...
        for i, param in enumerate(self.model.features.parameters()):
            if i < 4:  # First 4 layers (2 conv layers + 2 relu layers)
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Register hooks to capture activations for visualization
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture activations from specific layers"""
        def get_activation(name):
            def hook(model, input, output):
                self.feature_maps[name] = output.detach()
            return hook
        
        # First conv layer (general features)
        self.model.features[0].register_forward_hook(get_activation('conv1'))
        
        # Middle conv layer (3rd block, more specific patterns)
        self.model.features[10].register_forward_hook(get_activation('conv3'))
        
        # Deepest conv layer (5th block, complex features)
        self.model.features[24].register_forward_hook(get_activation('conv5'))
    
    def forward(self, x):
        return self.model(x)

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
    
    return epoch_loss, accuracy, precision, recall, f1, all_predictions, all_labels

# Function to visualize feature maps
def visualize_feature_maps(model, data_loader, layer_name, num_features=8, save_path=None):
    """Visualize feature maps from a specific layer based on Zeiler & Fergus approach"""
    # Get a batch of images
    inputs, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    
    # Forward pass to get feature maps
    with torch.no_grad():
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
        # Normalize for better visualization
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        axes[i].imshow(feature_map, cmap='viridis')
        axes[i].set_title(f"Filter {i+1}")
        axes[i].axis('off')
    
    plt.suptitle(f"Feature Maps from {layer_name} (Zeiler & Fergus Visualization)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# Occlusion sensitivity visualization (Zeiler & Fergus, 2014)
def occlusion_sensitivity(model, image, label, grid_size=8, occlusion_value=0.5):
    """
    Implementation of occlusion sensitivity from Zeiler & Fergus paper.
    Systematically occludes different portions of the input image and 
    monitors changes in the classification score.
    """
    # Original classification score
    with torch.no_grad():
        original_output = model(image)
        original_score = torch.softmax(original_output, dim=1)[0, label].item()
    
    # Get image dimensions
    _, C, H, W = image.shape
    
    # Calculate occlusion patch size
    patch_h, patch_w = H // grid_size, W // grid_size
    
    # Create heatmap
    sensitivity_map = np.zeros((grid_size, grid_size))
    
    # Create occlusion patch
    occlusion_patch = torch.ones((C, patch_h, patch_w)) * occlusion_value
    
    # Iterate through grid
    for i in range(grid_size):
        for j in range(grid_size):
            # Create occluded image
            occluded_image = image.clone()
            
            # Apply occlusion
            h_start, w_start = i * patch_h, j * patch_w
            h_end, w_end = min(h_start + patch_h, H), min(w_start + patch_w, W)
            
            occluded_image[0, :, h_start:h_end, w_start:w_end] = occlusion_patch[:, :h_end-h_start, :w_end-w_start]
            
            # Get prediction on occluded image
            with torch.no_grad():
                output = model(occluded_image)
                score = torch.softmax(output, dim=1)[0, label].item()
            
            # Calculate sensitivity (drop in confidence)
            sensitivity_map[i, j] = original_score - score
    
    return sensitivity_map

def visualize_occlusion_map(image, sensitivity_map, class_name, save_path=None):
    """Visualize occlusion sensitivity map overlaid on the original image"""
    # Convert image to numpy and denormalize
    img = inverse_normalize(image.cpu().squeeze(0))
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    
    # Resize sensitivity map to match image size
    heatmap = cv2.resize(sensitivity_map, (img.shape[1], img.shape[0]))
    
    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay heatmap on image
    overlay = 0.6 * img + 0.4 * heatmap_colored
    
    # Create figure with original image and overlay
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"Original Image ({class_name})")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='hot')
    plt.title("Occlusion Sensitivity")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis('off')
    
    plt.suptitle("Zeiler & Fergus Occlusion Sensitivity Visualization")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def main():
    # Create output directory for visualizations
    os.makedirs('vgg16_visualizations', exist_ok=True)
    
    # Load the dataset
    print("Loading dataset...")
    train_dataset = datasets.ImageFolder(root='train', transform=train_transform)

    # Get class names
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")

    # Split the training dataset into training and validation sets
    total_size = len(train_dataset)
    val_size = int(total_size * VALID_SPLIT)
    train_size = total_size - val_size
    
    # Create a fixed set of indices for reproducibility
    indices = list(range(total_size))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create train subset with train transforms
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    
    # Create validation dataset with test transforms
    val_dataset = datasets.ImageFolder(root='train', transform=test_transform)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")

    # Initialize the VGG16 fine-tuning model
    model = VGG16FineTuning(num_classes=NUM_CLASSES).to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

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
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

    # Calculate and save training time
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Save training time to file
    with open('vgg16_fine_tuned_train_time.txt', 'w') as f:
        f.write(f"{train_time:.2f}")

    # Save the model
    torch.save(model.state_dict(), 'vgg16_fine_tuned_model.pth')
    print("Model saved as vgg16_fine_tuned_model.pth")

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
    plt.savefig('vgg16_fine_tuning_curves.png')
    plt.close()

    # Final evaluation on validation set
    val_loss, val_acc, val_precision, val_recall, val_f1, all_preds, all_labels = evaluate(model, val_loader, criterion, device)
    
    print("\nFinal Model Performance on Validation Set:")
    print(f"Accuracy: {val_acc:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(cm)
    
    # Print per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds)
    
    print("\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1 Score: {f1[i]:.4f}")
        print(f"  Support: {support[i]}")
    
    # Visualize feature maps from different layers (Zeiler & Fergus approach)
    print("\nVisualizing feature maps...")
    
    # Get a representative image from each class for visualization
    class_images = {}
    class_labels = {}
    
    # Get one image from each class
    for class_idx, class_name in enumerate(class_names):
        for inputs, labels in val_loader:
            batch_class_indices = (labels == class_idx).nonzero(as_tuple=True)[0]
            if len(batch_class_indices) > 0:
                idx = batch_class_indices[0]
                class_images[class_name] = inputs[idx:idx+1].to(device)
                class_labels[class_name] = labels[idx].item()
                break
    
    # Visualize feature maps for each class
    for class_name, image in class_images.items():
        print(f"Visualizing features for class: {class_name}")
        
        # Forward pass to get feature maps
        with torch.no_grad():
            _ = model(image)
        
        # Visualize feature maps from different layers
        visualize_feature_maps(model, val_loader, 'conv1', 
                              save_path=f'vgg16_visualizations/vgg16_finetuning_{class_name}_conv1_features.png')
        visualize_feature_maps(model, val_loader, 'conv3', 
                              save_path=f'vgg16_visualizations/vgg16_finetuning_{class_name}_conv3_features.png')
        visualize_feature_maps(model, val_loader, 'conv5', 
                              save_path=f'vgg16_visualizations/vgg16_finetuning_{class_name}_conv5_features.png')
        
        # Occlusion sensitivity visualization (Zeiler & Fergus)
        sensitivity_map = occlusion_sensitivity(model, image, class_labels[class_name], grid_size=16)
        visualize_occlusion_map(image, sensitivity_map, class_name,
                               save_path=f'vgg16_visualizations/vgg16_finetuning_{class_name}_occlusion_sensitivity.png')

if __name__ == "__main__":
    main() 