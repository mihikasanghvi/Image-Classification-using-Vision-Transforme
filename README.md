# Image-Classification-using-Vision-Transformer
# Model Explanation:
1) Software Requirements:
- PyTorch: The code utilizes PyTorch for deep learning tasks.
- Transformers Library: It uses the `transformers` library for working with pre-trained 
transformer models, specifically ViT.
- TorchVision: This library is employed for image transformations and handling datasets.
- Matplotlib: Used for plotting confusion matrix and ROC curve.
- NumPy: For numerical operations.
  
2) Hardware Requirements:
- The code is designed to run on a machine with GPU support (100% GPU usage). The model 
is trained on a MAC using Jupyter Notebook.

# Model Architecture:
- The code imports the ViT model (`google/vit-base-patch16-224`) and its image processor from the `transformers` library.
- A custom classification head (`CustomHead`) is defined for the binary classification task, and it replaces the original classification head in the ViT model.
The choice of the Vision Transformer (ViT) model architecture, specifically `google/vit-basepatch16-224`, is motivated by its success in various computer vision tasks. ViT has shown 
strong performance in image classification, capturing long-range dependencies in images through self-attention mechanisms. The patch-based approach allows the model to process images globally, making it suitable for a diverse range of visual recognition tasks. The decision to replace the classification head with a custom head (`CustomHead`) tailored for a binary classification task is essential. This modification ensures that the model aligns with the specific requirements of the dataset, improving its ability to distinguish between 
the two classes.
For hyperparameters, the choice of the Adam optimizer with a learning rate of 1e-4 is a common starting point for fine-tuning tasks. This learning rate is chosen to strike a balance between convergence speed and stability during training. The number of training epochs (5) is a reasonable initial setting, considering computational resources and potential overfitting concerns.

# Data Processing:
- The code sets up data directories for training and testing images (`train_dir` and `test_dir`).
- Custom transformations are defined using `transforms.Compose`, including resizing images 
to (224, 224) pixels and converting them to tensors.
- DataLoaders are created for both training and testing sets using `create_dataloaders` 
function, which uses `datasets.ImageFolder` from TorchVision.

# Training:
- The model is trained using CrossEntropyLoss as the criterion and Adam optimizer with a learning rate of 1e-4.
- The training loop runs for 5 epochs. Each epoch consists of iterating through the training DataLoader, calculating and backpropagating the loss, and updating the model parameters.
  
# Evaluation:
- After each epoch, the model is evaluated on the test set to calculate validation accuracy.
- The model is set to evaluation mode using `model.eval()` and the accuracy is printed.

# Model Evaluation and Plots:
- The saved model is loaded for evaluation.
- The code then evaluates the model on a validation set (`valid_data`) and calculates 
accuracy. While accuracy provides a general overview of model performance, incorporating 
additional metrics like precision, recall, and F1 score can offer a more nuanced 
understanding. Precision measures the accuracy of positive predictions, recall assesses the 
model's ability to capture all relevant instances, and F1 score balances both precision and 
recall.
Integrating these metrics into the evaluation process provides insights into the model's 
performance on specific aspects of the binary classification task. It helps identify potential 
biases or weaknesses, offering a more comprehensive assessment beyond overall accuracy.
- A confusion matrix and ROC curve are generated using `sklearn.metrics` and plotted using 
Matplotlib.
