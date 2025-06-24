# IMDB Sentiment Classification with LSTM

## Project Overview

This project aims to classify movie reviews from the IMDB dataset as either positive or negative using a Long Short-Term Memory (LSTM) neural network. The process involves data loading, preprocessing, tokenization, building a vocabulary, encoding and padding the reviews, and finally training and evaluating an LSTM model for sentiment classification.

## Approach

The selected approach utilizes an LSTM network, which is well-suited for sequential data like text. The steps taken are as follows:

1.  **Data Loading and Exploration**: The IMDB dataset, containing movie reviews and their corresponding sentiment labels, is loaded into a pandas DataFrame.
2.  **Preprocessing**: A preprocessing function is applied to the reviews to remove HTML tags (`<br />`), special characters, convert text to lowercase, and tokenize the reviews using a basic English tokenizer. This helps in standardizing the text and reducing noise.
3.  **Vocabulary Creation and Encoding**: A vocabulary is built from the tokenized reviews, and each word is assigned a unique index. The reviews are then encoded into sequences of these indices. This converts the text data into a numerical format suitable for model input.
4.  **Padding**: The encoded sequences are padded to a fixed length to ensure uniform input size for the LSTM model. This is necessary because LSTMs require fixed-size inputs.
5.  **Dataset and DataLoader**: Custom PyTorch Dataset and DataLoader objects are created to handle the data efficiently during training and evaluation. DataLoaders provide iterable access to the dataset in batches, which is crucial for efficient training.
6.  **Model Definition**: An LSTM-based classifier model (`LSTMClassifier`) is defined with an embedding layer, LSTM layer, and a fully connected output layer. LSTMs are chosen for their ability to capture long-term dependencies in sequential data like text, which is important for understanding context in reviews. The embedding layer converts word indices into dense vectors, the LSTM layer processes these sequences, and the fully connected layer produces the final sentiment prediction.
7.  **Model Training**: The model is trained using the Adam optimizer and BCEWithLogitsLoss criterion. Adam is an adaptive learning rate optimization algorithm that is generally effective for training deep neural networks. BCEWithLogitsLoss is suitable for binary classification tasks like sentiment analysis. A learning rate scheduler is used to adjust the learning rate during training, which can help in achieving better convergence and preventing overfitting.
8.  **Gradient Clipping**: `clip_grad_norm_` is used to prevent exploding gradients during training. This is a common technique in training recurrent neural networks to avoid large updates to the model weights that can destabilize training.
9.  **Evaluation**: The trained model is evaluated on a separate test set to assess its performance using accuracy as the metric.

## Results

After training the model for 20 epochs, the following results were obtained:

*   **Training Accuracy**: The model achieved a training accuracy of approximately 90.12%.
*   **Validation Accuracy**: The model achieved a validation accuracy of approximately 86.03%.

These results indicate that the LSTM model is capable of classifying movie reviews with reasonable accuracy on this dataset.
