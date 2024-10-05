# Genre Classification Neural Network


## Overview
This project aims to classify music genres using a neural network. The model is built to take numerical feature values related to music tracks and predict their respective genres.


## Dataset
- The dataset used in this project was downloaded from [Kaggle](https://www.kaggle.com/).
- The dataset contains various audio features, including:
  - BPM (Beats Per Minute)
  - Danceability
  - Valence
  - Energy
  - Acousticness
  - Instrumentalness
  - Liveness
  - Speechiness
- The dataset does not contain a pre-defined genre column; genres are derived from the numerical feature values using a custom classification approach.


## Technologies Used
- Python
- Pandas for data manipulation
- Keras for building the neural network
- Scikit-learn for data preprocessing and model evaluation


## Installation
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your_username/genre-classification.git
   cd genre-classification

2. Install the required packages
   ```bash
   pip install pandas keras scikit-learn


## Usage
  1. Prepare your dataset as a CSV file.
  2. Update the FilePath variable in the dataProcessing function with the path to your dataset.
  3. Run the dataProcessing function to preprocess the dataset and add the genre column.
  4. Train the model using the preprocessed data:
            history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2)
  5. Evaluate the model on test data and analyze the results.
     

# Model Architecture
  - The neural network consists of:
      > Input layer
      > Three hidden layers with ReLU activation
      > Output layer with Softmax activation for multi-class classification


## Contact
For questions or feedback, please reach out at:
- **Email**: spyde40@gmail.com
- **GitHub**: [kprajval](https://github.com/kprajval)

   
