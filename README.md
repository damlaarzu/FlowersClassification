# 🌸 Flower Classification Project

## Project Overview
This project implements and compares three different convolutional neural network (CNN) models for classifying images of flowers into five categories: **daisy**, **dandelion**, **rose**, **sunflower**, and **tulip**. It demonstrates the effectiveness of different deep learning approaches for image classification tasks.

## Dataset
The dataset consists of flower images divided into five classes. The images were split into training and validation sets using an **80/20 split**. **Data augmentation** techniques were applied during training to improve model generalization.

## Models Implemented

You can reach trained models from here:
https://drive.google.com/drive/folders/1_0wB5WZ8y7rQvLDUEM_sPEEE4CNT3kcu?usp=sharing

### Model 1: Custom CNN
A basic convolutional neural network implemented from scratch with the following architecture:
- 5 convolutional blocks (each with convolution, ReLU activation, and max pooling)
- Fully connected layers with dropout for regularization
- Trained from scratch on the flower dataset

### Model 2: VGG16 Feature Extractor
This model uses the VGG16 architecture pre-trained on ImageNet as a **feature extractor**:
- The convolutional layers of VGG16 were **frozen** (weights not updated during training)
- The classifier was replaced with a custom classifier for the 5 flower classes
- Only the new classifier layers were trained

### Model 3: VGG16 Fine-Tuning
This model takes transfer learning further by **fine-tuning** the VGG16 model:
- Only the first convolutional block was frozen
- The remaining layers were fine-tuned during training
- The classifier was replaced with a custom classifier for the 5 flower classes

## Training Methodology
For all models, the following training settings were used:
- **Loss function**: Cross-entropy loss
- **Optimizer**: Adam (learning rate = 0.001 for Custom CNN, 0.0001 for transfer learning models)
- **Batch size**: 32 (16 for VGG16 Feature Extractor)
- **Data augmentation**: Random flips, rotations, and normalization

Additional techniques for the VGG16 Feature Extractor model to address class imbalance:
- Class-balanced loss function
- Weighted random sampling
- Mixup data augmentation
- Enhanced data transformations
- Batch normalization
- Early stopping based on balanced accuracy

## Visualization Techniques
Several visualization techniques based on the **Zeiler & Fergus approach** were implemented:
1. **Feature map visualization**: Displays activations from different convolutional layers
2. **Occlusion sensitivity maps**: Shows which parts of the image most influence the classification
3. **Class activation mapping**: Highlights the regions in the image that contribute most to the classification

## Results and Analysis

### Performance Metrics

| Model                   | Accuracy | Precision | Recall  | F1 Score | Inference Time (s) |
|------------------------|----------|-----------|---------|----------|-------------------|
| Custom CNN              | 0.9425   | 0.9427    | 0.9425  | 0.9423   | 11.99             |
| VGG16 Feature Extractor | 0.7804   | 0.7948    | 0.7804  | 0.7766   | 28.54             |
| VGG16 Fine-Tuned        | 0.9767   | 0.9771    | 0.9767  | 0.9767   | 26.68             |

### Key Findings
1. **VGG16 Fine-Tuned** model achieved the best performance across all metrics, showing the power of transfer learning with fine-tuning.
2. **Custom CNN** performed surprisingly well, achieving over 94% accuracy despite being trained from scratch with fewer parameters.
3. **VGG16 Feature Extractor** had the lowest performance, suggesting that using a pre-trained model as a feature extractor without fine-tuning may not be optimal for domain-specific images.
4. The **VGG16 Feature Extractor** initially struggled with class imbalance, mostly predicting images as *dandelion*. This was addressed with class-balanced loss and weighted sampling.
5. Inference time was faster for **Custom CNN (11.99s)** compared to the VGG16 models (26–28s), showing a trade-off between accuracy and computational efficiency.

## Conclusion
This project successfully implemented and compared three CNN-based approaches for flower classification. The **VGG16 Fine-Tuned** model achieved the best performance with **97.67% accuracy**, demonstrating that transfer learning with fine-tuning is highly effective for this task.

The comprehensive evaluation, including confusion matrices and per-class metrics, provides valuable insights into model behavior and performance. The visualization techniques implemented also offer qualitative understanding of how these models learn and make predictions.




# Flower Classification with CNN Models

This project implements and compares three different CNN models for flower image classification:

1. **Custom CNN** - A basic CNN model built from scratch
2. **VGG16 Feature Extractor** - Using pretrained VGG16 as a feature extractor
3. **VGG16 Fine-Tuning** - Fine-tuning VGG16 model by freezing early layers and training later layers

## Dataset

The dataset consists of flower images divided into five classes:
- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

## Project Structure

- `flower_cnn.py` - Implementation of custom CNN model
- `vgg16_feature_extractor.py` - Implementation of VGG16 feature extraction model
- `vgg16_fine_tuning.py` - Implementation of VGG16 fine-tuning model
- `model_comparison.py` - Script to evaluate and compare all three models
- `requirements.txt` - Required Python packages

## Setup and Running

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/macOS
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Prepare the dataset:
   - Place flower images in a `train` directory with subfolders for each class
   
5. Train the models:
   ```
   python flower_cnn.py
   python vgg16_feature_extractor.py 
   python vgg16_fine_tuning.py
   ```
   
6. Compare model performance:
   ```
   python model_comparison.py
   ```

## Results

The VGG16 Fine-Tuned model achieved the best performance with 97.67% accuracy, followed by the Custom CNN (94.25%) and VGG16 Feature Extractor (78.04%).

## Visualization Techniques

The project implements various visualization techniques based on the Zeiler & Fergus approach:
- Feature map visualization
- Occlusion sensitivity maps
- Class activation mapping
