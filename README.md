# Flower Classifier

This project focuses on building an image classifier to accurately identify different categories of flowers. I utilized pre-trained neural network models and modify them for the task, ensuring high accuracy and robust performance.

## Preprocessing

The image data was preprocessed to enhance the performance of the neural networks:
- **Resizing:** All images were resized to a uniform dimension.
- **Normalization:** Pixel values were normalized to assist in network training.
- **Augmentation:** To increase the robustness of the model, the dataset was augmented with random transformations such as rotation, flipping, and zooming.

## Models Used

Two pre-trained models were modified and used for flower classification:
- **VGG11**
- **AlexNet**

For both models, the last fully connected layer was updated to have 102 output units corresponding to the number of flower categories. The pre-trained weights of earlier layers were kept frozen.

## Results

The models achieved the following accuracy:
- **VGG11:** 75%
- **AlexNet:** 82%

The superior performance of AlexNet is likely due to its deeper architecture, which facilitates more complex feature extraction.

## Conclusion

The classifier demonstrates promising results in flower image classification. Future improvements could include experimenting with other pre-trained models, tuning hyperparameters, and expanding the dataset.

## Repository Contents
- `train.py`: Python script to train the models using any given dataset.
- `predict.py`: Python script for making predictions on new images.
- `cat_to_name.json`: Contains the mappings from category labels to flower names.






