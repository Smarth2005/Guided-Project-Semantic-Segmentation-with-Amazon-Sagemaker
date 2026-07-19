# Semantic Segmentation with Amazon SageMaker

Coursera Guided Project completed as part of the UCS654 – Predictive Analytics using Statistics course at Thapar Institute of Engineering and Technology (TIET).

![Course](https://img.shields.io/badge/Course-UCS654-blue)
![Platform](https://img.shields.io/badge/AWS-SageMaker-orange)
![Framework](https://img.shields.io/badge/PyTorch-EE4C2C)
![Dataset](https://img.shields.io/badge/Dataset-IIIT--Oxford%20Pets-success)
![Type](https://img.shields.io/badge/Coursera-Guided%20Project-0056D2)

## 1. Project Overview

This project demonstrates an end-to-end **semantic segmentation** workflow using **Amazon SageMaker**. It covers dataset preprocessing, model training, deployment of a real-time inference endpoint, and prediction on unseen images using the **IIIT-Oxford Pets Dataset**.

## 2. Implementation

To accommodate the unavailability of GPU instances, the original pipeline was adapted to use a lightweight **MobileNetV3-Large** architecture optimized for **CPU (`ml.m5`)** instances.

A custom **PyTorch** training script (`train.py`) was implemented with `drop_last=True` to ensure stable training using small batch sizes on CPU hardware.

## 3. Input / Output
* **Input:** Raw RGB Image (JPG/PNG).
* **Output:** 4-Class Segmentation Mask (Values: 0=Background, 1=Foreground, 2=Pet, 3=Boundary).

## 4. Deployment
* The trained model was deployed as a **real-time Amazon SageMaker Endpoint** for inference. The endpoint was deleted after testing to minimize AWS usage costs.

## 5. Results
The trained model successfully segmented pet images with clear distinction between the animal (Class 2) and the background, verified via visual inspection of the prediction masks.

![Predicted Segmentation Mask](images/my_segmentation_result.jpeg)

## 6. Resources

- The final trained model and processed data backup can be accessed here: [Open Link](https://drive.google.com/drive/u/0/folders/13teBlVBQjj6R_XFBl0FXj4yXdziI3jsA)
- **Platform Access:** This project was built and executed on the Amazon SageMaker platform: [https://console.aws.amazon.com/sagemaker/](https://console.aws.amazon.com/sagemaker/)



