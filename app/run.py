import json
import plotly
import pandas as pd

import sys
sys.path.append("..")

from flask import Flask
from flask import render_template, request
from joblib import load
from sqlalchemy import create_engine
from plot_data import wrangle_data

from models.train_classifier import tokenize, DocLength, NerExtractor


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message_with_category', engine)

# load model
model = load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    graphs = wrangle_data.generate_plots(df)
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result_1=list(classification_results.items())[0:12],
        classification_result_2=list(classification_results.items())[12:24],
        classification_result_3=list(classification_results.items())[24:36]
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
