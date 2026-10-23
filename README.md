This repository contains code for training and explainations, extra figures, and additional model statistics for Explaining Basketball Shot Classifications with Grad-CAM: An Investigation of Shortcut Reliance with Spatiotemporal Correlations.

**CONTENTS:**

*training/*
- figures: additional figures detailing the model's training and validation performance, as well as a confusion matrix created from the test dataset
- basketballtraining.ipynb: the code used to train the model, run on Kaggle Notebooks
- history.json: the mean and per-class loss and accuracy of each epoch
    - format: 
    [
        [mean training loss, mean training accuracy, [per-class training loss], [per class training accuracy], mean validation loss, mean validation accuracy, [per-class validation loss], [per-class validation accuracy]] 
    for each epoch]

*explanation/*
- figures: additional figures that add context to the explanation of the model
- manual_gradcam_analysis: each clip used in the visual Grad-CAM analysis
- basketballexplanations.ipynb: the code used to generate explanations on the trained model, also run on Kaggle Notebooks