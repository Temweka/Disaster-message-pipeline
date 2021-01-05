# Disaster message pipeline App
## Project Overview
This project is part of the Udacity Data Science Programme. The objection of this project is to create a web app that allows an emergency worker to input a new message and get classification results in several disaster categories, such that the message is directed to the correct agency. 

## File Descriptions
`process_data.py`: This code takes as its input csv files containing message data and message categories (labels), and creates an SQLite database containing a merged and cleaned version of this data.
`train_classifier.py`: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.
`ETL Pipeline Preparation.ipynb`: The code and analysis contained in this Jupyter notebook was used in the development of process_data.py. process_data.py effectively automates this notebook.
`ML Pipeline Preparation.ipynb`: The code and analysis contained in this Jupyter notebook was used in the development of train_classifier.py. In particular, it contains the analysis used to tune the ML model and determine which algorithm to use. train_classifier.py effectively automates the model fitting process contained in this notebook.
`data`: This folder contains sample messages and categories datasets in csv format.
`app`: This folder contains all of the files necessary to run and render the web app.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Libraries
This project was written in HTML and Python 3, and requires the following Python packages: 
1. json
2. plotly
3. pandas 
4. nltk 
5. flask 
6. sklearn 
7. sqlalchemy 
8. numpy 
9. re 
10. pickle
