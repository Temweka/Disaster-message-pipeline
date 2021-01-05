import sys
import sqlalchemy as db
import pandas as pd
import numpy as np
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import pickle


def load_data(database_filepath):
    '''
    description (function): loads dataframe from database
    inputs:
        database_filepath (str): path to dataframe file
    output:
        X (dataframe): merged dataframe of the two files
        Y (dataframe): merged dataframe of the two files
        category_names (list): merged dataframe of the two files
    '''
    engine = db.create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_msg_ctg_tbl',engine)
    X = df['message']  
    Y = df.iloc[:, 4:] 
    Y['related']=Y['related'].map(lambda x: 1 if x == 2 else x)
    category_names = list(Y.columns.values)
    return X, Y, category_names


def tokenize(text):    
    '''
    description (function): 
        tokenizes text
    inputs:
        text (list): list of text messages
    output:
        clean_token (list): tokenized text
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    words = [w for w in tokens if w not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()

    clean_token = [lemmatizer.lemmatize(tok) for tok in words]

    return clean_token


def build_model():
    '''
    description (function): 
        builds machine learning model using Scikit learn Pipeline

    output:
        pipeline: ML model 
    '''    
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer = tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf',  MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    description (function): 
        predicts and evaluates the performance of the model on unseen data
    inputs:
        model: prepared model
        X_test (df): dataframe of feature variables
        Y_test (df): dataframe of target variable
        category_names (list): list of category names
        
    output:
        Classification_report: prints the precision, recall and f1-score of model, and the accuracy of model
       
    '''
    
    y_pred = model.predict(X_test)

    # print classification report
    print(classification_report(Y_test.values, y_pred, target_names=category_names))

    # print accuracy score
    print('Accuracy: {}'.format((Y_test.values == y_pred).mean()))


def save_model(model, model_filepath):
    '''
    description (function): 
        save model
    inputs:
        model: the model
        model_filepath (str): path to where the model is to be saved

    '''
    with open (model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()