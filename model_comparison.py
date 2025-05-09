import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tabulate

# Import model definitions
from flower_cnn import FlowerCNN
from vgg16_feature_extractor import VGG16FeatureExtractor
from vgg16_fine_tuning import VGG16FineTuning

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
IMAGE_SIZE = 224
NUM_CLASSES = 5
BATCH_SIZE = 32
NUM_WORKERS = 0

# Create output directory
os.makedirs('comparison_results', exist_ok=True)

# Define data transformations for evaluation
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(model_type):
    """Load a specific model for evaluation"""
    try:
        if model_type == "Custom CNN":
            print(f"  Trying to load Custom CNN model...")
            model = FlowerCNN(num_classes=NUM_CLASSES).to(device)
            model.load_state_dict(torch.load('flower_cnn_model.pth', map_location=device))
            print(f"  Custom CNN model loaded successfully!")
            return model
        elif model_type == "VGG16 Feature Extractor":
            print(f"  Trying to load VGG16 Feature Extractor model...")
            model = VGG16FeatureExtractor(num_classes=NUM_CLASSES).to(device)
            # First try the best model, then fall back to final if best doesn't exist
            try:
                print(f"  Attempting to load vgg16_feature_extractor_best.pth...")
                model.load_state_dict(torch.load('vgg16_feature_extractor_best.pth', map_location=device))
                print(f"  Loaded vgg16_feature_extractor_best.pth successfully!")
            except Exception as e:
                print(f"  Error loading best model: {str(e)}")
                print(f"  Attempting to load vgg16_feature_extractor_final.pth...")
                model.load_state_dict(torch.load('vgg16_feature_extractor_final.pth', map_location=device))
                print(f"  Loaded vgg16_feature_extractor_final.pth successfully!")
            return model
        elif model_type == "VGG16 Fine-Tuned":
            print(f"  Trying to load VGG16 Fine-Tuned model...")
            model = VGG16FineTuning(num_classes=NUM_CLASSES).to(device)
            # Try multiple possible filenames
            try:
                print(f"  Attempting to load vgg16_fine_tuned_model.pth...")
                model.load_state_dict(torch.load('vgg16_fine_tuned_model.pth', map_location=device))
                print(f"  Loaded vgg16_fine_tuned_model.pth successfully!")
            except Exception as e1:
                print(f"  Error loading vgg16_fine_tuned_model.pth: {str(e1)}")
                try:
                    print(f"  Attempting to load vgg16_fine_tuning_model.pth...")
                    model.load_state_dict(torch.load('vgg16_fine_tuning_model.pth', map_location=device))
                    print(f"  Loaded vgg16_fine_tuning_model.pth successfully!")
                except Exception as e2:
                    print(f"  Error loading vgg16_fine_tuning_model.pth: {str(e2)}")
                    print(f"  Could not find a saved model for {model_type}. Please train the model first.")
                    return None
            return model
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        print(f"  Error loading {model_type} model: {str(e)}")
        return None

def evaluate_model(model, model_type, test_loader):
    """Evaluate a model on the test set and return performance metrics"""
    if model is None:
        return {
            "Model": model_type,
            "Accuracy": "N/A",
            "Precision": "N/A",
            "Recall": "N/A",
            "F1 Score": "N/A",
            "Train Time (s)": "N/A"
        }
    
    model.eval()
    
    # Use cross entropy loss for evaluation
    criterion = nn.CrossEntropyLoss()
    
    # Time the inference
    start_time = time.time()
    
    # Collect predictions
    all_predictions = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate inference time
    inference_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    # Calculate per-class metrics
    class_precision, class_recall, class_f1, support = precision_recall_fscore_support(all_labels, all_predictions)
    
    # Get the training time from the file if it exists
    train_time = "Unknown"
    
    if model_type == "Custom CNN":
        train_time_file = "flower_cnn_train_time.txt"
    elif model_type == "VGG16 Feature Extractor":
        train_time_file = "vgg16_feature_extractor_train_time.txt"
    elif model_type == "VGG16 Fine-Tuned":
        train_time_file = "vgg16_fine_tuned_train_time.txt"
    
    # Try to read training time from file
    try:
        with open(train_time_file, 'r') as f:
            train_time = f.read().strip()
    except:
        train_time = "Unknown"
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Return metrics dictionary
    return {
        "Model": model_type,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall, 
        "F1 Score": f1,
        "Train Time (s)": train_time,
        "Inference Time (s)": inference_time,
        "Confusion Matrix": cm,
        "Class Precision": class_precision,
        "Class Recall": class_recall,
        "Class F1": class_f1,
        "Support": support
    }

def compare_models():
    """Compare performance of all models"""
    # Load test data from the train directory since it has the proper class structure
    # In a real scenario, you would use separate validation/test data
    test_dataset = datasets.ImageFolder(root='train', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Get class names
    class_names = test_dataset.classes
    print(f"Classes: {class_names}")
    
    # Define models to evaluate
    model_types = ["Custom CNN", "VGG16 Feature Extractor", "VGG16 Fine-Tuned"]
    
    # Evaluate each model
    results = []
    detailed_results = {}
    
    for model_type in model_types:
        print(f"Evaluating {model_type}...")
        
        try:
            model = load_model(model_type)
            if model is None:
                print(f"  Model {model_type} could not be loaded, skipping evaluation.")
                results.append({
                    "Model": model_type,
                    "Accuracy": "N/A",
                    "Precision": "N/A",
                    "Recall": "N/A",
                    "F1 Score": "N/A",
                    "Train Time (s)": "N/A",
                    "Inference Time (s)": "N/A"
                })
                continue
                
            metrics = evaluate_model(model, model_type, test_loader)
            results.append(metrics)
            detailed_results[model_type] = metrics
            
            if isinstance(metrics['Accuracy'], float):
                print(f"  Accuracy: {metrics['Accuracy']:.4f}")
                print(f"  Precision: {metrics['Precision']:.4f}")
                print(f"  Recall: {metrics['Recall']:.4f}")
                print(f"  F1 Score: {metrics['F1 Score']:.4f}")
            else:
                print(f"  Accuracy: {metrics['Accuracy']}")
                print(f"  Precision: {metrics['Precision']}")
                print(f"  Recall: {metrics['Recall']}")
                print(f"  F1 Score: {metrics['F1 Score']}")
                
            print(f"  Train Time: {metrics['Train Time (s)']}")
            
            if isinstance(metrics['Inference Time (s)'], float):
                print(f"  Inference Time: {metrics['Inference Time (s)']:.4f} seconds\n")
            else:
                print(f"  Inference Time: {metrics['Inference Time (s)']}\n")
            
        except Exception as e:
            print(f"Error evaluating {model_type}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                "Model": model_type,
                "Accuracy": "Error",
                "Precision": "Error",
                "Recall": "Error",
                "F1 Score": "Error",
                "Train Time (s)": "Error",
                "Inference Time (s)": "Error"
            })
    
    print(f"Completed evaluation of {len(results)} models")
    print(f"Results: {results}")
    
    # Create performance table
    performance_df = pd.DataFrame([
        {
            "Model": r["Model"],
            "Accuracy": f"{r['Accuracy']:.4f}" if isinstance(r['Accuracy'], float) else r['Accuracy'],
            "Precision": f"{r['Precision']:.4f}" if isinstance(r['Precision'], float) else r['Precision'],
            "Recall": f"{r['Recall']:.4f}" if isinstance(r['Recall'], float) else r['Recall'],
            "F1 Score": f"{r['F1 Score']:.4f}" if isinstance(r['F1 Score'], float) else r['F1 Score'],
            "Train Time (s)": r["Train Time (s)"],
            "Inference Time (s)": f"{r['Inference Time (s)']:.4f}" if isinstance(r['Inference Time (s)'], float) else r['Inference Time (s)']
        } for r in results
    ])
    
    # Print performance table
    print("\nModel Performance Comparison:")
    print(tabulate.tabulate(performance_df, headers="keys", tablefmt="grid"))
    
    # Create comparison plots
    plot_model_comparison(results, class_names)
    
    # Generate HTML report
    generate_html_report(performance_df, detailed_results, class_names)
    
    # Save performance table as CSV
    performance_df.to_csv("comparison_results/model_performance_comparison.csv", index=False)
    
    return performance_df

def plot_model_comparison(results, class_names):
    """Create comparison plots for models"""
    
    # Extract valid results (where we have numeric metrics)
    valid_results = [r for r in results if isinstance(r['Accuracy'], float)]
    
    if not valid_results:
        print("No valid results to plot")
        return
    
    # Plot accuracy, precision, recall, F1 score comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    models = [r['Model'] for r in valid_results]
    
    plt.figure(figsize=(12, 8))
    
    # Extract metrics for each model
    metric_values = {
        metric: [r[metric] for r in valid_results]
        for metric in metrics
    }
    
    # Create DataFrame for grouped bar plot
    comparison_df = pd.DataFrame({
        'Model': models * len(metrics),
        'Metric': [metric for metric in metrics for _ in range(len(models))],
        'Value': [metric_values[metric][i] for metric in metrics for i in range(len(models))]
    })
    
    # Create grouped bar plot
    sns.barplot(x='Model', y='Value', hue='Metric', data=comparison_df)
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('comparison_results/model_performance_comparison.png')
    plt.close()
    
    # Plot confusion matrices for each model
    for r in valid_results:
        if 'Confusion Matrix' in r:
            plt.figure(figsize=(10, 8))
            cm = r['Confusion Matrix']
            df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
            sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - {r["Model"]}')
            plt.tight_layout()
            plt.savefig(f'comparison_results/confusion_matrix_{r["Model"].replace(" ", "_").lower()}.png')
            plt.close()

def generate_html_report(performance_df, detailed_results, class_names):
    """Generate a comprehensive HTML report"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flower Classification Model Comparison</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 12px; text-align: left; border: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .model-section {{ margin-bottom: 30px; padding: 15px; border: 1px solid #eee; border-radius: 5px; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            .metric-definition {{ background-color: #f9f9f9; padding: 10px; border-left: 4px solid #2980b9; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Flower Classification Model Comparison</h1>
        
        <div class="metric-definition">
            <h3>Metric Definitions:</h3>
            <p><strong>Accuracy:</strong> The proportion of correctly classified images out of all images.</p>
            <p><strong>Precision:</strong> The proportion of true positive predictions out of all positive predictions.</p>
            <p><strong>Recall:</strong> The proportion of true positive predictions out of all actual positives.</p>
            <p><strong>F1 Score:</strong> The harmonic mean of precision and recall.</p>
            <p><strong>Train Time:</strong> Time taken to train the model (in seconds).</p>
            <p><strong>Inference Time:</strong> Time taken to make predictions on the test set (in seconds).</p>
        </div>
        
        <h2>Overall Performance Comparison</h2>
        <table>
            <tr>
                {" ".join([f"<th>{col}</th>" for col in performance_df.columns])}
            </tr>
            {"".join([f"<tr>{' '.join([f'<td>{cell}</td>' for cell in row])}</tr>" for row in performance_df.values.tolist()])}
        </table>
        
        <img src="model_performance_comparison.png" alt="Performance Comparison Chart" />
        
    """
    
    # Add detailed section for each model
    for model_type, metrics in detailed_results.items():
        if isinstance(metrics['Accuracy'], float):  # Only include valid results
            html_content += f"""
            <div class="model-section">
                <h2>{model_type}</h2>
                <h3>Overall Metrics:</h3>
                <ul>
                    <li><strong>Accuracy:</strong> {metrics['Accuracy']:.4f}</li>
                    <li><strong>Precision:</strong> {metrics['Precision']:.4f}</li>
                    <li><strong>Recall:</strong> {metrics['Recall']:.4f}</li>
                    <li><strong>F1 Score:</strong> {metrics['F1 Score']:.4f}</li>
                    <li><strong>Train Time:</strong> {metrics['Train Time (s)']}</li>
                    <li><strong>Inference Time:</strong> {metrics['Inference Time (s)']:.4f} seconds</li>
                </ul>
                
                <h3>Per-Class Metrics:</h3>
                <table>
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1 Score</th>
                        <th>Support</th>
                    </tr>
            """
            
            # Add per-class metrics
            for i, class_name in enumerate(class_names):
                html_content += f"""
                    <tr>
                        <td>{class_name}</td>
                        <td>{metrics['Class Precision'][i]:.4f}</td>
                        <td>{metrics['Class Recall'][i]:.4f}</td>
                        <td>{metrics['Class F1'][i]:.4f}</td>
                        <td>{metrics['Support'][i]}</td>
                    </tr>
                """
                
            html_content += f"""
                </table>
                
                <h3>Confusion Matrix:</h3>
                <img src="confusion_matrix_{model_type.replace(' ', '_').lower()}.png" alt="{model_type} Confusion Matrix" />
            </div>
            """
    
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML report to file
    with open("comparison_results/model_comparison_report.html", "w") as f:
        f.write(html_content)
    
    print("HTML report generated: comparison_results/model_comparison_report.html")

if __name__ == "__main__":
    # Add installation check for tabulate
    try:
        import tabulate
    except ImportError:
        print("Installing tabulate package...")
        import subprocess
        subprocess.call(["pip", "install", "tabulate"])
        import tabulate
    
    # Run the comparison
    compare_models() 