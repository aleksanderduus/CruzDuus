# Hotdog / Not Hotdog Classification

## 1. Introduction

The objective of this project was to classify images into two classes: `hotdog` and `not hotdog`. The dataset contains hotdog images and non-hotdog images from several ImageNet categories, including food, frankfurter, chili-dog, pets, furniture, and people.

The project compares convolutional neural networks trained from scratch with a pretrained convolutional neural network used through transfer learning.

## 2. Data and preprocessing

All input images have different resolutions and aspect ratios. To create batches with a common shape, the images were resized to 128 x 128 pixels for the CNN experiments.

The training pipeline used:

- Resize to 128 x 128
- Random horizontal flipping
- Random rotation of up to 10 degrees
- Random brightness, contrast, and saturation changes
- Conversion to tensors
- Normalization with mean `(0.5, 0.5, 0.5)` and standard deviation `(0.5, 0.5, 0.5)`

The test pipeline used resizing, tensor conversion, and the same normalization, but no random augmentation. This ensures that evaluation is deterministic.

For ResNet18, the images were resized to 224 x 224 pixels and normalized using the ImageNet mean and standard deviation, as expected by the pretrained model.

## 3. CNN trained from scratch

The first model was a small CNN with three convolutional layers. The number of channels increased from 3 to 16, 32, and 64. Each convolutional layer was followed by a ReLU activation and max pooling. The classifier consisted of a fully connected layer with dropout followed by a two-class output layer.

The model was trained using cross-entropy loss and the Adam optimizer with a learning rate of 0.001.

The final recorded result was:

| Model | Training accuracy | Test accuracy |
|---|---:|---:|
| Basic CNN | 83.29% | 78.79% |

The difference between training and test accuracy indicates some overfitting. However, the model still learned useful visual features and performed clearly above random guessing.

## 4. Regularization and batch normalization

A second CNN was designed to reduce overfitting. It used:

- Batch normalization after each convolution
- Adaptive average pooling
- Dropout with probability 0.5
- Adam with weight decay of `1e-4`
- Data augmentation
- Early stopping

Batch normalization stabilizes the distribution of activations during training. It can make optimization easier and often allows the network to train more reliably. Dropout and weight decay reduce the tendency of the model to memorize individual training examples.

The regularized model achieved a best recorded test accuracy of approximately 77.23% before early stopping. In this experiment, batch normalization and stronger regularization did not improve the final test accuracy compared with the basic CNN. This is an important result: regularization can reduce overfitting without necessarily increasing accuracy, especially when the model is already relatively small or the dataset contains difficult examples.

## 5. Deeper CNN

A deeper CNN was also tested by adding a fourth convolutional block with 128 channels. The model included batch normalization, dropout, adaptive average pooling, and weight decay.

The deeper model achieved:

| Model | Training accuracy | Test accuracy |
|---|---:|---:|
| Deeper CNN | 83.29% | 78.79% |

The additional convolutional block did not produce a major improvement. Increasing depth gives the model more capacity to learn hierarchical features, but it also increases the risk of overfitting and does not guarantee better generalization.

## 6. Transfer learning with ResNet18

ResNet18 was used as a transfer learning model. ResNet18 is itself a convolutional neural network. It contains convolutional residual blocks and was pretrained on ImageNet.

All pretrained parameters were frozen, and the final fully connected layer was replaced with a classifier for two classes. The new classifier consisted of dropout followed by a linear layer with two outputs. Only this final classifier was trained using Adam.

The recorded result was:

| Model | Training accuracy | Test accuracy |
|---|---:|---:|
| ResNet18 transfer learning | 90.52% | 92.27% |

ResNet18 performed substantially better than the CNNs trained from scratch. The pretrained convolutional layers already contain useful low-level and mid-level visual features, such as edges, textures, shapes, and object parts. This is especially valuable when the available dataset is not large enough to train a deep network from scratch effectively.

## 7. Cross-validation

To evaluate the stability of the transfer learning result, stratified five-fold cross-validation was performed on the training set. Stratification preserved the class distribution in each fold. The separate test set was not used to create the folds.

The validation accuracies were:

| Fold | Validation accuracy |
|---:|---:|
| 1 | 93.90% |
| 2 | 92.44% |
| 3 | 92.18% |
| 4 | 94.87% |
| 5 | 92.18% |

The mean validation accuracy was **93.11%**, with a standard deviation of **1.09 percentage points**.

The relatively small standard deviation indicates that the ResNet18 result is reasonably stable across different training and validation splits.

## 8. Comparison of models

| Model | Main characteristics | Test accuracy |
|---|---|---:|
| Basic CNN | Three convolutional layers, Adam | 78.79% |
| Regularized CNN | Batch normalization, dropout, augmentation, weight decay | 77.23% |
| Deeper CNN | Four convolutional blocks, batch normalization, dropout | 78.79% |
| ResNet18 | ImageNet pretraining, frozen convolutional backbone | 92.27% |

The strongest model was ResNet18 with transfer learning. The results suggest that pretrained features were more important than simply increasing the depth of a small CNN trained from scratch.

## 9. Discussion

The experiments show that increasing model complexity alone is not sufficient to obtain better performance. The deeper CNN had more feature extraction layers, but its test accuracy remained similar to the basic CNN. This indicates that the main limitation was not simply the number of layers.

Batch normalization and regularization helped control the training process, but the regularized model did not outperform the baseline in this run. The random augmentation may also have made the training task harder, while the dataset size and visual variation limited the achievable performance of the small CNNs.

Transfer learning was substantially more effective. ResNet18 could use representations learned from a much larger dataset, which allowed it to generalize better to the hotdog classification task.

## 10. Limitations and remaining analysis

The current notebook does not yet contain a systematic list of misclassified test images. It also does not yet contain saliency maps or SmoothGrad visualizations. These should be added before submitting the final PDF report.

The test set was used for repeated comparison during the experiments. For a stricter scientific evaluation, model selection should be based on a validation set or cross-validation, and the test set should only be evaluated once at the end.

## 11. Use of generative AI

ChatGPT was used as a programming assistant during the project. It helped with debugging file paths and tensor shapes, suggesting model variations, explaining convolutional neural networks and transfer learning, and organizing parts of the report.

The project decisions, interpretation of the results, and understanding of the underlying machine learning methods were based on the student's existing background in artificial intelligence and independent evaluation of the experiments. ChatGPT was used as support, not as a replacement for the student's technical work or judgment.

## 12. Conclusion

The best-performing approach was transfer learning with ResNet18. It achieved 92.27% test accuracy, while five-fold cross-validation produced a mean validation accuracy of 93.11% with a standard deviation of 1.09 percentage points.

The results demonstrate that pretrained convolutional features can provide a substantial advantage over a small CNN trained from scratch. Further work should include a detailed error analysis and saliency/SmoothGrad visualizations to better understand which image regions influence the predictions.
