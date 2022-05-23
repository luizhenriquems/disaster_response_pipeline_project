import sys

from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import pickle


def load_data(database_filepath):
    """
    load_data
    Load the filepath and return the data
    
    Inputs: database_filepath - sqllite file storing clean dataset, created by process_Data.py
    
    Returns:
    X - input data which is message sent 
    y - categories which we need to predict 
    category_names
    """
    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    df = pd.read_sql_table('DisastersResponse', con=engine) 
    print(df.head())
    # X is input data which is message sent 
    # Y is 36 categories which we need to predict which are last 36 columns 
    X = df['message']
    y = df[df.columns[-36:]]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    '''
    tokenize
    Function for tokenizing string
    
    Inputs: Text string
    
    Returns: List of clean tokens
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    build_model
    Function for building pipeline and GridSearch
    
    Inputs: None
    
    Returns: Model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # Parameters set to this to speed up the model
    parameters = {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2],
    
    }
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=3)
    return model


def evaluate_model(model, X_test, y_test, category_names):
    '''
    evaluation_model
    Function for evaluating model through a classification report
    
    Inputs:   Model, features, labels to evaluate and a list of categories
    
    Returns: Classification report
    '''
    y_pred = model.predict(X_test)
    class_report = classification_report(y_test, y_pred, target_names=category_names)
    print(class_report)


def save_model(model, model_filepath):
    '''
    save_model
    Function for saving the model as picklefile
    
    Inputs: Model and filepath
    
    Returns: None.
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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