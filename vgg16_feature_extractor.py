import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import seaborn as sns
from collections import Counter
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True  # For reproducibility

# Define hyperparameters
BATCH_SIZE = 16  # Smaller batch size for better generalization
EPOCHS = 20  # More epochs for thorough learning
LEARNING_RATE = 0.0001  # Lower learning rate
IMAGE_SIZE = 224  # VGG16 expects 224x224 images
NUM_CLASSES = 5
VALID_SPLIT = 0.2  # 20% of the training data will be used for validation
NUM_WORKERS = 0  # Set to 0 to avoid multiprocessing issues

# Force CUDA
try:
    # Try to set the device to CUDA
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    # If CUDA is not available, use CPU
    device = torch.device("cpu")
    print(f"CUDA error: {e}")
    print("Using CPU instead.")

# Enhanced data transformations with stronger augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 30, IMAGE_SIZE + 30)),  # Resize larger for random crop
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),  # Random crop for more variation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),  # Add vertical flip
    transforms.RandomRotation(45),  # Increased rotation range
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Stronger color jittering
    transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10),  # Enhanced affine transformation
    transforms.RandomGrayscale(p=0.05),  # Occasionally convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)  # Random erasing for occlusion robustness
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the VGG16 Feature Extractor model with more adaptations for flower classification
class VGG16FeatureExtractor(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super(VGG16FeatureExtractor, self).__init__()
        
        # Load pretrained VGG16 model
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # Extract features (convolutional layers)
        self.features = vgg16.features
        
        # Unfreeze all layers for full fine-tuning
        # This will help the model adapt better to our specific dataset
        for param in self.features.parameters():
            param.requires_grad = True
            
        # Add batch normalization to features to stabilize training
        self.features_bn = nn.Sequential()
        for i, layer in enumerate(self.features):
            self.features_bn.add_module(str(i), layer)
            # Add batch norm after each conv layer
            if isinstance(layer, nn.Conv2d):
                self.features_bn.add_module(f"bn{i}", nn.BatchNorm2d(layer.out_channels))
        
        # Replace classifier with a fully customized one
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Keep the adaptive pooling
        
        # Improved classifier with higher dropout for regularization
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),  # Add batch norm for stable training
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),  # Add batch norm
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 256),  # Extra layer
            nn.BatchNorm1d(256),   # Add batch norm
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize the classifier weights for better convergence
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features using VGG16 convolutional layers with batch norm
        x = self.features_bn(x)
        
        # Apply adaptive pooling
        x = self.avgpool(x)
        
        # Flatten the features
        x = torch.flatten(x, 1)
        
        # Pass through our custom classifier
        x = self.classifier(x)
        
        return x

# Class-balanced loss function to handle class imbalance
class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_class, num_classes, beta=0.9999, gamma=2.0):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        
        # Calculate effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)
        
        # Calculate weights
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes
        weights = torch.tensor(weights).float().to(device)
        
        # Store cross entropy with weights
        self.cross_entropy = nn.CrossEntropyLoss(weight=weights, reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        return loss.mean()

# Training function with mixup and label smoothing
def train(model, train_loader, criterion, optimizer, device, mixup_alpha=0.2, label_smoothing=0.1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply mixup augmentation
        if np.random.random() < 0.7 and mixup_alpha > 0:  # Increased probability of mixup
            # Generate mixed inputs and targets
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(inputs.size(0)).to(device)
            mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
            
            # Label smoothing for one-hot encoded targets
            mixed_labels_a = torch.zeros(labels.size(0), NUM_CLASSES).to(device)
            mixed_labels_b = torch.zeros(labels.size(0), NUM_CLASSES).to(device)
            
            for i, label in enumerate(labels):
                mixed_labels_a[i, label] = 1.0 - label_smoothing
                mixed_labels_a[i, :] += label_smoothing / NUM_CLASSES
                
                mixed_labels_b[i, labels[index][i]] = 1.0 - label_smoothing
                mixed_labels_b[i, :] += label_smoothing / NUM_CLASSES
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed inputs
            outputs = model(mixed_inputs)
            
            # Calculate loss with mixup - using custom loss with one-hot labels
            loss = lam * custom_cross_entropy(outputs, mixed_labels_a) + (1 - lam) * custom_cross_entropy(outputs, mixed_labels_b)
            
            # For accuracy calculation, we'll use the primary label
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Class-wise accuracy for monitoring
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
                
        else:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Regular forward pass
            outputs = model(inputs)
            
            # Use CrossEntropyLoss directly
            loss = criterion(outputs, labels)
            
            # Statistics
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Class-wise accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
        
        # Backward and optimize
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    
    # Calculate class-wise accuracy
    class_accuracy = [class_correct[i] / max(class_total[i], 1) for i in range(NUM_CLASSES)]
    
    return epoch_loss, epoch_acc, class_accuracy

# Custom cross entropy for one-hot labels
def custom_cross_entropy(outputs, targets):
    log_softmax = torch.log_softmax(outputs, dim=1)
    loss = -torch.sum(targets * log_softmax, dim=1)
    return loss.mean()

# Evaluation function
def evaluate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Class-wise accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
    
    # Calculate metrics
    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    # Calculate class-wise accuracy
    class_accuracy = [class_correct[i] / max(class_total[i], 1) for i in range(NUM_CLASSES)]
    
    return epoch_loss, accuracy, precision, recall, f1, all_predictions, all_labels, class_accuracy

def main():
    # Create output directory for results
    os.makedirs('results', exist_ok=True)
    
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
    
    # Analyze class distribution
    train_labels = [train_dataset.targets[i] for i in train_indices]
    class_counts = Counter(train_labels)
    print("Class distribution in training set:")
    for class_idx, count in class_counts.items():
        print(f"  {class_names[class_idx]}: {count} samples")
    
    # Calculate highly effective class weights for extremely imbalanced data
    # Inverse frequency with a maximum cap to prevent extreme weights
    max_count = max(class_counts.values())
    class_weights = [min(max_count / max(class_counts[i], 1), 10.0) for i in range(NUM_CLASSES)]
    class_weights = torch.FloatTensor(class_weights).to(device)
    print("Class weights:", class_weights)
    
    # Create weighted sampler with square root technique for balanced sampling
    samples_per_class = [class_counts[i] for i in range(NUM_CLASSES)]
    effective_samples = [int(np.sqrt(count)) for count in samples_per_class]
    class_weights_sqrt = [max_count / max(count, 1) for count in effective_samples]
    sample_weights = [class_weights_sqrt[train_dataset.targets[i]] for i in train_indices]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_subset), replacement=True)
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")

    # Initialize the VGG16 feature extractor model with higher dropout
    model = VGG16FeatureExtractor(num_classes=NUM_CLASSES, dropout_rate=0.6).to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")

    # Use the Class-Balanced Loss for severe imbalance
    criterion = ClassBalancedLoss(samples_per_class, NUM_CLASSES, beta=0.9999, gamma=2.0)
    
    # Use AdamW optimizer with cosine learning rate schedule and weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4  # L2 regularization
    )
    
    # Cosine annealing learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # Restart every 5 epochs
        T_mult=2,  # Double the restart period after each restart
        eta_min=1e-6  # Minimum learning rate
    )

    # Training loop
    print("Starting training...")
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_f1 = 0.0
    best_model_path = 'vgg16_feature_extractor_best.pth'
    
    # For early stopping
    patience = 5
    counter = 0
    
    # Track class-wise accuracy
    train_class_accs = []
    val_class_accs = []
    
    # Record start time
    start_time = time.time()

    for epoch in range(EPOCHS):
        # Train
        train_loss, train_acc, train_class_acc = train(
            model, train_loader, criterion, optimizer, device, 
            mixup_alpha=0.3, label_smoothing=0.1
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_class_accs.append(train_class_acc)
        
        # Print class-wise training accuracy
        print(f"Epoch {epoch+1} - Class-wise training accuracy:")
        for i, acc in enumerate(train_class_acc):
            print(f"  {class_names[i]}: {acc:.4f}")
        
        # Evaluate
        val_loss, val_acc, val_precision, val_recall, val_f1, all_preds, all_labels, val_class_acc = evaluate(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_class_accs.append(val_class_acc)
        
        # Print class-wise validation accuracy
        print(f"Epoch {epoch+1} - Class-wise validation accuracy:")
        for i, acc in enumerate(val_class_acc):
            print(f"  {class_names[i]}: {acc:.4f}")
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Early stopping based on balanced accuracy (average of class-wise accuracies)
        balanced_acc = sum(val_class_acc) / len(val_class_acc)
        
        # Save best model based on balanced accuracy instead of F1
        if balanced_acc > best_val_f1:
            best_val_f1 = balanced_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch+1}: Saved new best model with balanced accuracy: {balanced_acc:.4f}")
            counter = 0  # Reset counter
        else:
            counter += 1
            
        # Apply early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Balanced Acc: {balanced_acc:.4f} | "
              f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

    # Calculate and save training time
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Save training time to file
    with open('vgg16_feature_extractor_train_time.txt', 'w') as f:
        f.write(f"{train_time:.2f}")

    # Save the final model
    torch.save(model.state_dict(), 'vgg16_feature_extractor_final.pth')
    print("Final model saved as vgg16_feature_extractor_final.pth")
    
    # Load the best model for final evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path))

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
    plt.savefig('results/vgg16_extractor_training_curves.png')
    plt.close()
    
    # Plot class-wise accuracy over epochs
    plt.figure(figsize=(15, 6))
    
    # Training accuracy
    plt.subplot(1, 2, 1)
    for i in range(NUM_CLASSES):
        class_acc_over_time = [epoch_acc[i] for epoch_acc in train_class_accs]
        plt.plot(class_acc_over_time, label=class_names[i])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Class-wise Training Accuracy')
    plt.legend()
    
    # Validation accuracy
    plt.subplot(1, 2, 2)
    for i in range(NUM_CLASSES):
        class_acc_over_time = [epoch_acc[i] for epoch_acc in val_class_accs]
        plt.plot(class_acc_over_time, label=class_names[i])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Class-wise Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/vgg16_extractor_class_wise_accuracy.png')
    plt.close()

    # Final evaluation on validation set
    val_loss, val_acc, val_precision, val_recall, val_f1, all_preds, all_labels, val_class_acc = evaluate(
        model, val_loader, criterion, device
    )
    
    balanced_acc = sum(val_class_acc) / len(val_class_acc)
    
    print("\nFinal Model Performance on Validation Set:")
    print(f"Accuracy: {val_acc:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1 Score: {val_f1:.4f}")
    
    # Compute and plot normalized confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot non-normalized confusion matrix
    plt.figure(figsize=(10, 8))
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/vgg16_extractor_confusion_matrix.png')
    plt.close()
    
    # Plot normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    df_cm_norm = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
    sns.heatmap(df_cm_norm, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/vgg16_extractor_confusion_matrix_normalized.png')
    plt.close()
    
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
        print(f"  Accuracy: {val_class_acc[i]:.4f}")
    
    # Plot per-class metrics
    metrics_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': val_class_acc
    }, index=class_names)
    
    plt.figure(figsize=(12, 6))
    metrics_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Per-class Performance Metrics')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/vgg16_extractor_per_class_metrics.png')
    plt.close()
    
    # Print prediction distribution
    pred_counts = Counter(all_preds)
    print("\nPrediction distribution:")
    for class_idx, count in sorted(pred_counts.items()):
        print(f"  {class_names[class_idx]}: {count} predictions ({count/len(all_preds)*100:.2f}%)")

if __name__ == "__main__":
    main() 