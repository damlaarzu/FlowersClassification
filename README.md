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

## License

This project is licensed under the MIT License - see the LICENSE file for details. 