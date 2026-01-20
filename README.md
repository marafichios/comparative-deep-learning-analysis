# Comparative Deep Learning Analysis: Images & Text

This project implements and evaluates various deep learning architectures for two distinct tasks: **Computer Vision (Image Classification)** and **Natural Language Processing (Sentiment Analysis)**. 

The goal was to analyze the impact of architectural choices, hyperparameter tuning, and data augmentation on model performance, culminating in 2 technical reports detailing the experimental methodology.

### Part 1: Image Classification
* **Datasets:** * `Imagebits`: 10-class general object dataset ($96\times96$ RGB).
    * `Land Patches`: 10-class satellite imagery dataset ($64\times64$ RGB) for land use classification.
* **Architectures:** * **MLP (Multi-Layer Perceptron):** Baseline fully connected networks.
    * **CNN (Convolutional Neural Networks):** Custom architectures exploring convolutional filters, pooling layers, and batch normalization.
* **Techniques:** * Data Augmentation using `Albumentations` (Random Rotate, Flip, Color Jitter).
    * Fine-tuning strategies for transfer learning between datasets.
    * Comparative analysis of hyperparameter configurations and learning rate schedulers.

### Part 2: Sentiment Analysis (Romanian Language)
* **Dataset:** `ro_sent` (17k+ Romanian reviews labeled Positive/Negative).
* **Architectures:**
    * **Simple RNN:**
    * **LSTM (Long Short-Term Memory):** Implemented Bi-directional LSTMs to capture long-range dependencies in text.
* **Pipeline:**
    * **Preprocessing:** Tokenization via `Spacy`, stop-word removal, and text normalization.
    * **Embeddings:** Integration of pre-trained **FastText** vectors for dense word representation.
    * **Augmentation:** Random swap/insert and back-translation techniques to balance class distribution.

## Results & Analysis
The project includes a comprehensive analysis of:
* **Overfitting mitigation:** The effects of Dropout, L2 Regularization, and Early Stopping.
* **Performance Metrics:** Accuracy, F1-Score, and Confusion Matrices for all models.
* **Ablation Studies:** Evaluating the specific impact of data augmentation on model generalization.
