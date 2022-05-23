import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load and merge data from csv files to a single dataframe
    
    Inputs: messages_filepath: The file path to the messages.csv file.
            categories_filepath: The file path to the categories.csv file.
            
    Returns: A pandas DataFrame containing both files.
    '''
    
    # Load and merge the datasets
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    df = messages.merge(categories, on='id', how='inner')
    
    return df 


def clean_data(df):
    '''
    clean_data
    Split and replace categories column and then drop the duplicates
    
    Inputs: loaded and merged pandas df
    
    Returns: pandas df cleaned 
    '''
    
    # Split categories into separate category columns
    # Create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(';', expand=True)
    # Select only the first row of the dataframe
    row = categories.iloc[0,:]
    # Extract a list of new column names for categories from the row created
    category_colnames = row.apply(lambda x: x[:-2])
    # Rename the columns of "Categories"
    categories.columns = category_colnames
        # Convert category values to just numbers 0 or 1.
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # Convert column from string to int
        categories[column] = categories[column].astype(int)
    categories.replace(2, 1, inplace=True)
    
    # Drop the original categories from `df`
    df.drop('categories', axis=1, inplace = True)
    
    # Concatenate the original dataframe with the just created
    df = pd.concat([df, categories], axis=1)
    
    #Drop duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    Saves DataFrame (df) to database path
    param
    df - pandas dataframe, cleaned by clean_data function
    database_filename - path of database file to store cleaned dataframe
    """
    name = 'sqlite:///' + database_filename
    engine = create_engine(name)
    df.to_sql('DisastersResponse', engine, index=False, if_exists='replace') 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()