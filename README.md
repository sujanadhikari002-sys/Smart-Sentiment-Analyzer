# Smart Sentiment Analyzer

## Description
This project is an NLP-based sentiment analysis system that classifies customer reviews as **Positive** or **Negative** using a trained neural network. The model is trained on an Amazon product review dataset and deployed using a Flask web application to provide real-time predictions.

## Dataset
The system uses the Amazon Product Review dataset, which contains reviews from multiple categories including Books, Electronics, DVD, and Kitchen products. The dataset includes pre-labelled positive and negative reviews and was reconstructed from a token-frequency format into readable text for training.

## Technologies
- Python  
- TensorFlow / Keras  
- Flask  
- HTML / Tailwind CSS  

## Model Details
A neural network model was developed using TensorFlow, consisting of Dense layers with ReLU activation and Dropout layers to reduce overfitting. The output layer uses a Sigmoid activation function for binary classification. The model achieved approximately **55% accuracy** on the test dataset.

## Features
- Real-time sentiment prediction  
- Displays confidence score  
- Simple and user-friendly web interface  

## How to Run
1. Install dependencies:
   pip install flask tensorflow  

2. Run the application:
   python app.py  

3. Open in browser:
   http://127.0.0.1:5000  

## Notes
The model performs better with clear and well-structured input text. It may struggle with spelling errors, informal language, or unseen vocabulary due to limitations in the dataset and model complexity.