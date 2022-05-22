# Disaster Response Pipeline Project

## Project Context

This project is part of the Data Science Nanodegree from Udacity and its regarding to a Machine Learning Pipeline
The first step of the project was to understand and extract, transform and load (ETL pipeline) the data provided by Figure Eight - the data was stored in a sqlite database. Then there was the construction of the model and posterior evaluation - the idea was to classify the messages into 36 categories. The final step was the initiation of a web page related to emergency situations worldwide.

## Used Libraries

nltk
sklearn
flask
plotly
pandas
sqlalchemy

## Dataset

The dataset was provided by Figure Eight and its splited into two files: messages.csv and categories.csv.

## Instructions

Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in the database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/


