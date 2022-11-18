from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import joblib
import language_tool_python
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline


def removeUnwantedChars(df):
    df['full_text'] = df['full_text'].replace(r'\n', ' ', regex=True).replace(
        r'\r', ' ', regex=True).replace(r'\\', ' ', regex=True)


def findLength(df):
    df['length'] = np.nan
    df['length'] = df['full_text'].str.len()


def findMistakes(df):

    df['errors'] = df.apply(lambda x: allErrors(
        x['full_text'], checker),  axis=1)


class readyData_transformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        removeUnwantedChars(X)
        findLength(X)
        findMistakes(X)
        return X

# Function for finding frequency of chars in text.


def frequencyChars(text, char):

    # Creates counter - dictionary of (char, number of occurences)
    count = Counter(text)

    char_occ = count.get(char)

    # In case char is not in text
    if char_occ is None:

        char_occ = 0

    # Returns frequency of char in text
    return char_occ/len(text)


def matchesByRules(matches, ruleIds):

    applicable = [None] * len(matches)
    numberApplicable = 0

    for match in matches:

        if match.ruleId in ruleIds:
            applicable[numberApplicable] = match
            numberApplicable += 1

    return applicable[:numberApplicable]


def matchesByIssueType(matches, issueTypes):

    applicable = [None] * len(matches)
    numberApplicable = 0

    for match in matches:

        if match.ruleIssueType in issueTypes:
            applicable[numberApplicable] = match
            numberApplicable += 1

    return applicable[:numberApplicable]


def matchesByCategory(matches, categories):

    applicable = [None] * len(matches)
    numberApplicable = 0

    for match in matches:

        if match.category in categories:
            applicable[numberApplicable] = match
            numberApplicable += 1

    return applicable[:numberApplicable]


class addFeatures_transformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X['F_period/Sentence_length'] = X.apply(
            lambda x: frequencyChars(x['full_text'], '.'),  axis=1)

        X['F_errors'] = X.apply(lambda x: len(x['errors'])/x['length'], axis=1)

        # criteria
        punctuation_category = ['PUNCTUATION']

        X['punctuation_errors'] = X.apply(lambda x: matchesByCategory(
            x['errors'], punctuation_category), axis=1)
        X['F_punctuation_errors'] = X.apply(lambda x: len(
            x['punctuation_errors'])/x['length'], axis=1)

        # criteria
        misspelling_ruleIssueType = ['misspelling']

        X['spelling_errors'] = X.apply(lambda x: matchesByIssueType(
            x['errors'], misspelling_ruleIssueType),  axis=1)
        X['F_spelling_errors'] = X.apply(lambda x: len(
            x['spelling_errors'])/x['length'], axis=1)

        # criteria
        casing_category = ['CASING']

        X['casing_errors'] = X.apply(lambda x: matchesByCategory(
            x['errors'], casing_category),  axis=1)
        X['F_casing_errors'] = X.apply(lambda x: len(
            x['casing_errors'])/x['length'], axis=1)

        grammar_ruleIssueType = ['grammar']

        X['grammar_errors'] = X.apply(lambda x: matchesByIssueType(
            x['errors'], grammar_ruleIssueType),  axis=1)
        X['F_grammar_errors'] = X.apply(lambda x: len(
            x['grammar_errors'])/x['length'], axis=1)

        return X


class removeTextFeaturesAndLength_transformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        columnsToDrop = ['length', 'full_text', 'errors', 'punctuation_errors',
                         'spelling_errors', 'casing_errors', 'grammar_errors']

        X.drop(columnsToDrop, axis=1, inplace=True)

        return X


cohesion_model = joblib.load('models/cohesion_model.joblib')
syntax_model = joblib.load('models/syntax_best.joblib')
vocabulary_model = joblib.load('models/vocabulary_best.joblib')
phraseology_model = joblib.load('models/phraseology_best.joblib')
grammar_model = joblib.load('models/grammar_best.joblib')
conventions_model = joblib.load('models/conventions_best.joblib')

checker = language_tool_python.LanguageTool(
    'en-US', config={'cacheSize': 1000, 'pipelineCaching': True, 'maxSpellingSuggestions': 0})


# Takes text (string) and instance of LanguageTool as input. Returns list of Match instances.
def allErrors(text, checker):

    return checker.check(text)


def preprocess(data):
    """
    Returns the features entered by the user in the web form. 

    To simplify, we set a bunch of default values. 
            For bools and ints, use the most frequent value
            For floats, use the median value

    Note that this represent some major assumptions that you'd 
    not want to make in real life. If you want to use default 
    values for some features then you'll have to think more 
    carefully about what they should be. 

    F.ex. if the user doesn't provide a value for BMI, 
    then one could use a value that makes more sense than 
    below. For example, the mean for the given gender would 
    at least be a bit more correct. 

    Having _dynamic defaults_ is important. And of course, if 
    relevant, getting some of the features without asking the user. 
    E.g. if the user is logged in and you can pull information 
    form a user profile. Or if you can compute or obtain the information 
    thorugh other means (e.g. user location if shared etc).
    """

    feature_values = {
        'full_text': ''
    }

    # Parse the form inputs and return the defaults updated with values entered.

    feature_values['full_text'] = data['full_text']

    return feature_values

#######
# Now we can predict with the trained model:
#######


def predict(data):
    """
    If debug, print various useful info to the terminal.
    """

    # Store the data in an array in the correct order:
    data = data.get('full_text')
    print(data)

    data = pd.DataFrame({'full_text': [data]})

    pipe = Pipeline(

        steps=[
            ('readyData', readyData_transformer()),
            ('addFeatures', addFeatures_transformer()),
            ('removeTextFeaturesAndLength',
             removeTextFeaturesAndLength_transformer()),
            ('scaler', MinMaxScaler())
        ]

    )

    pipe.fit_transform(data)
    # NB: In this case we didn't do any preprocessing of the data before
    # training our random forest model (see the notebool `nbs/1.0-asl-train_model.ipynb`).
    # If you plan to feed the training data through a preprocessing pipeline in your
    # own work, make sure you do the same to the data entered by the user before
    # predicting with the trained model. This can be achieved by saving an entire
    # sckikit-learn pipeline, for example using joblib as in the notebook.

    pred = cohesion_model.predict(data)
    if(pred < 0):
        pred = 0

    return pred


def postprocess(prediction):
    """
    Apply postprocessing to the prediction. E.g. validate the output value, add
    additional information etc. 
    """

    pred = prediction

    # Validate. As an example, if the output is an int, check that it is positive.
    try:
        int(pred[0]) > 0
    except:
        pass

    # Kan ikke få lavere enn 0.
    pred = float(pred)
    if pred < 0:
        pred = 0

    pred = str(pred)
    # Make strings
    pred = str(pred[0])

    # Return
    return_dict = {'pred': pred, 'uncertainty': '100%'}

    return return_dict
