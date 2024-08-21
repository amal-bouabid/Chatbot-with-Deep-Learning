# Chatbot with Deep Learning
This project implements a simple chatbot using a Deep Neural Network (DNN) in TensorFlow/Keras. The chatbot is designed to understand and respond to user queries based on a predefined set of intents.

## Description
The chatbot is built using Natural Language Processing (NLP) techniques, enabling it to interpret user inputs and generate appropriate responses. The model is trained on a JSON dataset containing various intents, each with corresponding patterns (questions) and responses. After training, the chatbot can predict the most suitable tag for a given user input and retrieve a relevant response.

## Dependencies
To run this project, you'll need the following Python libraries:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- TensorFlow
##  Installation
- Install the required libraries using pip.
- Download the necessary NLTK data for stopwords, tokenization, and lemmatization.
- Place your dataset (e.g., KB.json) in the appropriate directory.
## Usage
After setting up the environment and dataset, you can train and evaluate the chatbot model. The model is designed with three layers:

- An input layer with 128 neurons and ReLU activation.
- A hidden layer with 64 neurons and ReLU activation.
- An output layer with softmax activation, corresponding to the number of intents.
Once trained, the model can predict the intent of user inputs and generate responses accordingly.

## Model Training and Evaluation
The model is trained using the sparse_categorical_crossentropy loss function and optimized with adam. It is evaluated based on accuracy and loss metrics, with an expected accuracy of around X% on the test set.

## Testing the Chatbot
You can test the chatbot by inputting a question, and the model will predict the intent and return a corresponding response. The chatbot selects a random response from the possible options for the predicted intent.
