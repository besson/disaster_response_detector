import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Load and consolidate both data sources
    INPUT
        messages_filepath - file path for reading message data, str
        categories_filepath - file path for reading category data, str

    OUTPUT
        df - a dataset consolidating input data sources, pd.DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return messages.merge(categories, on=['id'])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up the raw dataset:
       - Split disaster categories into separate columns
       - Convert category data types
       - Remove unneeded columns
       - Remove duplicated rows
    INPUT
        df - a raw dataset, pd.DataFrame

    OUTPUT
        df - a cleaned dataset, pd.DataFrame
    """
    # Split categories
    categories = df['categories'].str.split(pat=';', expand=True)

    # Define category names
    row = categories.head(1).values[0]
    category_colnames = list(map(lambda x: x.split('-')[0], row))
    categories.columns = category_colnames

    # Convert category types
    for column in categories:
        categories[column] = pd.Series(categories[column]).astype(str) \
            .str.split('-').str[1] \
            .astype(int)

    # Add categories to main DataFrame
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df: pd.DataFrame, database_filename: str):
    """
    Persists data set into a database
    INPUT
        df - cleaned dataset, pd.DataFrame

    OUTPUT
        None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('message_with_category', engine, index=False)


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
