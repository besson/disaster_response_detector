from nltk import word_tokenize, WordNetLemmatizer
from plotly.graph_objs import Scatter, Bar
from wordcloud import WordCloud


def generate_plots(df):
    """
    Generate plot objected to be rendered int the dashboard:
        - Bar chart to plot distribution of genre
        - Bar chart to plot distribution of disaster category types
        - Word cloud to plot frequency of word in message content
    INPUT
        df - training set, pd.DataFrame
    OUTPUT
        graphs - list of plotly objects, List
    """
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # melt dataframe
    df1 = df.melt(id_vars=['id', 'message', 'original', 'genre'], var_name='category', value_name='active')

    # Graph 2 - Distribution of category types
    category_counts = df1[df1.active == 1].groupby('category').agg({'message': 'count'}) \
                                          .reset_index().sort_values(by='message', ascending=True)
    category_names = category_counts['category'].values

    # Graph 3 - Wordcloud of sample of messages (Sample of 100 messages)
    words = df.sample(100)['message'].apply(_tokenize).values
    words = [word for word_list in words for word in word_list]

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_counts['message'],
                    y=category_names,
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Distribution of Disaster category types',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                },
                'margin': dict(l=150, r=15, pad=10)
            }
        }
    ]
    wc = _plotly_wordcloud(' '.join(words))
    graphs.append(wc)

    return graphs


def _tokenize(text):
    """
    Tokenize words from input sentences
    INPUT
        text - message content, str
    OUTPUT
        cleaned tokens - cleaned tokens after tokenization phase, List
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def _plotly_wordcloud(text):
    """
    Word cloud plot. Based on: https://github.com/PrashantSaikia/Wordcloud-in-Plotly
    INPUT
        text - message content, str
    OUTPUT
        chart - word cloud chart, plotly objects
    """
    wc = WordCloud(max_words=200,
                   max_font_size=40,
                   min_font_size=2,
                   min_word_length=3)
    wc.generate(text)

    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []

    for (word, freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # get the positions
    x = []
    y = []
    for i in position_list:
        x.append(i[0])
        y.append(i[1])

    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i * 100)
    new_freq_list

    wc_plot_data = {
        'data': [
            Scatter(
                x=x,
                y=y,
                textfont=dict(size=new_freq_list,
                              color=color_list),
                hoverinfo='text',
                hovertext=['{0}: {1}'.format(w, f) for w, f in zip(word_list, freq_list)],
                mode='text',
                text=word_list
            )
        ],
        'layout': {
            'title': 'Message: Word cloud',
            'xaxis': {'showgrid': False,
                      'showticklabels': False,
                      'zeroline': False},
            'yaxis': {'showgrid': False,
                      'showticklabels': False,
                      'zeroline': False},
        }
    }
    return wc_plot_data
