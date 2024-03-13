# Flower-Classifier
##Preprocessing of Image Data
To prepare the image data for optimal use in neural networks, the data was preprocessed. This involved resizing the images to a uniform size, normalizing the pixel values, and augmenting the dataset by applying random transformations such as rotation, flipping, and zooming.

##Modified Pre-Trained Models
To achieve high accuracy in classification, two pre-trained models were modified - VGG11 and AlexNet. The models were fine-tuned by updating the last fully connected layer to have 102 output units for the number of flower categories. The rest of the model was kept frozen to retain the pre-trained weights.

Results
The modified VGG11 model achieved an accuracy of 75%, while the modified AlexNet model achieved an accuracy of 82%. The higher accuracy of the AlexNet model was attributed to its deeper architecture, allowing for more complex feature extraction.

Conclusion
The developed image classifier shows promising results for accurately classifying flower images. Further optimization can be done by exploring other pre-trained models, adjusting hyperparameters, and increasing the amount of data.

Repository Contents
This repository contains the following files:

README.md: This file, providing an overview of the project.

train.py: Python Script to evaluate the models on any given data.

predict.py: Python script to make predictions on new images.

cat_to_name.json: a json file containing labels of categories.


