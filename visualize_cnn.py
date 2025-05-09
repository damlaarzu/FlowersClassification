import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import cv2
from flower_cnn import FlowerCNN, IMAGE_SIZE

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data transformations
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

def load_model(model_path='flower_cnn_model.pth', num_classes=5):
    """Load a trained FlowerCNN model"""
    model = FlowerCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def get_feature_maps(model, image):
    """Get feature maps from all layers for a single image"""
    # Forward pass
    with torch.no_grad():
        _ = model(image)
    
    # Return feature maps stored in the model
    return model.feature_maps

def visualize_feature_maps(feature_maps, layer_name, num_features=8, save_path=None):
    """Visualize feature maps from a specific layer"""
    # Get feature maps for the specified layer
    features = feature_maps[layer_name].cpu().numpy()[0]  # Get first image in batch
    
    # Create a figure
    fig, axes = plt.subplots(2, num_features//2, figsize=(15, 6))
    axes = axes.flatten()
    
    # Display a subset of feature maps
    for i in range(min(num_features, features.shape[0])):
        feature = features[i]
        # Normalize for better visualization
        feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
        axes[i].imshow(feature, cmap='viridis')
        axes[i].set_title(f"Filter {i+1}")
        axes[i].axis('off')
    
    plt.suptitle(f"Feature Maps from {layer_name}")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def occlusion_sensitivity(model, image, label, grid_size=8, occlusion_value=0.5):
    """
    Implementation of occlusion sensitivity from Zeiler & Fergus paper.
    Systematically occludes different portions of the input image and 
    monitors changes in the classification score.
    """
    # Original classification score
    with torch.no_grad():
        original_output = model(image)
        original_score = F.softmax(original_output, dim=1)[0, label].item()
    
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
                score = F.softmax(output, dim=1)[0, label].item()
            
            # Calculate sensitivity (drop in confidence)
            sensitivity_map[i, j] = original_score - score
    
    return sensitivity_map

def visualize_occlusion_map(image, sensitivity_map, save_path=None):
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
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay heatmap on image
    overlay = 0.6 * img + 0.4 * heatmap
    
    # Create figure with original image and overlay
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Occlusion Sensitivity")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

class SaveFeatures():
    """Hook to save feature maps during forward pass"""
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = None
        
    def hook_fn(self, module, input, output):
        self.features = output
        
    def close(self):
        self.hook.remove()

# Simplified DeconvNet implementation
class DeconvNetSimple:
    """Simplified Deconvolutional Network for visualizing CNN features (Zeiler & Fergus, 2014)"""
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.feature_maps = {}
        
    def register_hooks(self):
        """Register hooks to capture activations"""
        def hook_fn(name):
            def hook(module, input, output):
                self.feature_maps[name] = output
            return hook
        
        # Register hooks for the layers we want to visualize
        self.hooks.append(self.model.conv1.register_forward_hook(hook_fn('conv1')))
        self.hooks.append(self.model.conv3.register_forward_hook(hook_fn('conv3')))
        self.hooks.append(self.model.conv5.register_forward_hook(hook_fn('conv5')))
        
    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def get_activations(self, image, layer_name):
        """Get activations for a specific layer"""
        # Register hooks
        self.register_hooks()
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(image)
            
        # Get activations
        activations = self.feature_maps[layer_name]
        
        # Remove hooks
        self.remove_hooks()
        
        return activations
        
    def generate_visualization(self, image, layer_name, filter_idx=0):
        """Generate visualization for a specific filter in a layer using gradient ascent"""
        # Create a copy of the image that requires gradients
        img = image.clone().detach().requires_grad_(True)
        
        # Optimizer
        optimizer = torch.optim.Adam([img], lr=0.1)
        
        for _ in range(20):  # Number of optimization steps
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            activations = self.get_activations(img, layer_name)
            
            # Target is to maximize the activation of the specific filter
            loss = -activations[0, filter_idx].mean()
            
            # Backward pass
            loss.backward()
            
            # Update image
            optimizer.step()
            
            # Apply regularization
            with torch.no_grad():
                # Normalize image for better visualization
                img.data = torch.clamp(img.data, 0, 1)
                
        # Return the optimized image
        return img.detach()

def visualize_deconvnet(model, image, layer_name, num_filters=8, save_path=None):
    """Visualize features using a simplified DeconvNet approach (Zeiler & Fergus, 2014)"""
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    axes = axes.flatten()
    
    # Create DeconvNet
    deconvnet = DeconvNetSimple(model)
    
    # Get activations for the layer
    activations = deconvnet.get_activations(image, layer_name)
    
    # Original image for reference
    orig_img = inverse_normalize(image.cpu().squeeze(0)).permute(1, 2, 0).numpy()
    orig_img = np.clip(orig_img, 0, 1)
    
    # For each filter, create a visualization
    for i in range(min(num_filters, activations.shape[1])):
        # Get the feature map for this filter
        feature_map = activations[0, i].cpu().numpy()
        
        # Normalize for visualization
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        # Resize to match image size for overlay
        resized_map = cv2.resize(feature_map, (orig_img.shape[1], orig_img.shape[0]))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * resized_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Overlay on original image
        overlay = 0.7 * orig_img + 0.3 * heatmap
        
        # Display
        axes[i].imshow(overlay)
        axes[i].set_title(f"Filter {i+1}")
        axes[i].axis('off')
    
    plt.suptitle(f"DeconvNet Features from {layer_name} (Zeiler & Fergus)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def deconvolution_visualization(model, layer_name, filter_idx, steps=30, lr=0.1):
    """
    Implementation of feature visualization via optimization.
    Finds an input image that maximizes the activation of a specific filter.
    """
    # Create a hook to get the feature maps
    if layer_name == 'conv1':
        target_layer = model.conv1
    elif layer_name == 'conv3':
        target_layer = model.conv3
    elif layer_name == 'conv5':
        target_layer = model.conv5
    else:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    # Register hook
    hook = SaveFeatures(target_layer)
    
    # Start with random noise
    optimizing_img = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, requires_grad=True, device=device)
    
    # Optimize the image
    optimizer = torch.optim.Adam([optimizing_img], lr=lr)
    
    for i in range(steps):
        optimizer.zero_grad()
        
        # Forward pass
        _ = model(optimizing_img)
        
        # Get activations from the hook
        activations = hook.features
        
        # Make sure filter_idx is valid
        if filter_idx >= activations.shape[1]:
            filter_idx = 0
            
        # Define loss as negative mean activation of target filter
        loss = -activations[0, filter_idx].mean()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Apply regularization for smoother images
        if i % 5 == 0:
            with torch.no_grad():
                # Normalize image for better visualization
                optimizing_img.data = torch.clamp(optimizing_img.data, -2, 2)
    
    # Remove the hook
    hook.close()
    
    # Return the optimized image
    return optimizing_img.detach()

def visualize_filters(model, layer_name, num_filters=8, steps=30, save_path=None):
    """Visualize what patterns activate specific filters in a layer"""
    plt.figure(figsize=(15, 6))
    
    for i in range(num_filters):
        # Generate an image that maximizes the activation of this filter
        optimized_img = deconvolution_visualization(model, layer_name, i, steps)
        
        # Normalize for visualization
        img = inverse_normalize(optimized_img.cpu().squeeze(0))
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        plt.subplot(2, num_filters//2, i+1)
        plt.imshow(img)
        plt.title(f"Filter {i+1}")
        plt.axis('off')
    
    plt.suptitle(f"Patterns that activate filters in {layer_name}")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def main():
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    
    try:
        # Load the model
        model = load_model()
        
        # Load a test image
        test_dataset = datasets.ImageFolder(root='train', transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        
        # Get a sample image
        images, labels = next(iter(test_loader))
        image = images.to(device)
        label = labels.item()
        class_name = test_dataset.classes[label]
        
        print(f"Visualizing for image of class: {class_name}")
        
        # 1. Visualize feature maps from different layers
        feature_maps = get_feature_maps(model, image)
        
        for layer_name in ['conv1', 'conv3', 'conv5']:
            visualize_feature_maps(
                feature_maps, 
                layer_name, 
                num_features=8, 
                save_path=f'visualizations/flowercnn_{class_name}_{layer_name}_features.png'
            )
        
        # 2. DeconvNet visualization (from Zeiler & Fergus)
        print("Generating DeconvNet visualizations...")
        for layer_name in ['conv1', 'conv3', 'conv5']:
            visualize_deconvnet(
                model,
                image,
                layer_name,
                num_filters=8,
                save_path=f'visualizations/flowercnn_{class_name}_deconvnet_{layer_name}.png'
            )
        
        # 3. Occlusion sensitivity visualization (from Zeiler & Fergus)
        sensitivity_map = occlusion_sensitivity(model, image, label, grid_size=16)
        visualize_occlusion_map(
            image, 
            sensitivity_map, 
            save_path=f'visualizations/flowercnn_{class_name}_occlusion_sensitivity.png'
        )
        
        # 4. Filter visualization through optimization - skip if errors occur
        try:
            for layer_name in ['conv1', 'conv3', 'conv5']:
                visualize_filters(
                    model, 
                    layer_name, 
                    num_filters=8, 
                    save_path=f'visualizations/flowercnn_{class_name}_filter_patterns_{layer_name}.png'
                )
        except RuntimeError as e:
            print(f"Error in filter visualization: {e}")
            print("Skipping filter visualization and continuing with other visualizations.")
        
        print("Visualization complete. Results saved in 'visualizations' directory.")
    
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
