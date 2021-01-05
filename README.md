# Disaster message pipeline App
## Project Overview
This project is part of the Udacity Data Science Programme. The objection of this project is to create a web app that allows an emergency worker to input a new message and get classification results in several disaster categories, such that the message is directed to the correct agency. 

## File Descriptions
`data`: This folder contains sample messages and categories datasets in csv format; and code file to preprocess the data.
    `process_data.py`: Processes the data by loading the two datasets, merging, cleaning and storing it in a SQLite database
    `ETL Pipeline Preparation.ipynb`: The code and analysis contained in this Jupyter notebook was used in the development of `process_data.py`. 

`models`: This folder contains saved model; and the code file to create said model.
    `train_classifier.py`: Creates the machine learning pipeline, to classify the data loaded from the SQLite database. 
    `ML Pipeline Preparation.ipynb`: The code and analysis contained in this Jupyter notebook was used in the development of `train_classifier.py`. 

`app`: This folder contains all of the files necessary to run and render the web app.
    `run.py`: USed to open the web page

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
