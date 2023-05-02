# Machine Learning Playground
## Overview

Machine Learning Playground is an interactive web-based application designed to provide users with hands-on experience in various machine learning algorithms. The primary goal of this project is to allow users to gain a deeper understanding of how these algorithms work and to observe the impact of different parameters on the results.

The application is built using Dash, a Python framework for building web applications, and Scikit-learn, a popular machine learning library.
## How it Works

Machine Learning Playground allows users to choose from a variety of datasets and machine learning algorithms. Users can then adjust parameters and see how the algorithm performs on the selected dataset. The application provides visualization of the data and the model's decision boundary, as well as performance metrics for the model.

The supported machine learning algorithms include:

    Support Vector Classifier
    Logistic Regression
    Linear Regression
    KMeans
    Orthogonal Projections
    Principle Component Analysis

## Running the Application

To run the Machine Learning Playground, follow these steps:

1. Clone the repository to your local machine:
    ```console
    git clone https://github.com/branisk/ml_playground.git
    ```

2. Change to the project directory:
    ```console
    cd ml_playground
    ```

3. Create a virtual environment and activate it:
    ```console
    python -m venv venv
    source venv/bin/activate
    ```

4. Install the required packages:
    ```console
    pip install -r requirements.txt
    ```
    
5. Run the application:
    ```console
    python index.py
    ```
    
6. Open your browser and navigate to http://127.0.0.1:8050/ to access the Machine Learning Playground.

# To-Do List

- [ ] Add more algorithms (e.g., K-Nearest Neighbors, Decision Trees, Random Forest, etc.)
- [ ] Include more datasets and data preprocessing options
- [ ] Improve the user interface and user experience
- [ ] Implement cross-validation for model evaluation
- [ ] Add more performance metrics and visualizations
- [ ] Provide explanations and guidance for each algorithm and parameter

Feel free to contribute to the project by submitting pull requests or opening issues on the GitHub repository.
