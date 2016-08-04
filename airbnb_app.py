import flask
app = flask.Flask(__name__)

#---------this app route is just for testing purposes....--------#
@app.route("/hello")
def hello():
    print "Hello World!!"
    return "Hello World!"

#-------- MODEL GOES HERE -----------#

import numpy as np
import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# from sklearn.cross_validation import train_test_split, cross_val_score
# from sklearn.metrics import roc_curve
# from sklearn.neighbors import KNeighborsClassifier



# df = pd.read_csv('speed_dating_user_attributes.csv')

# df_v2 = df.drop(df.columns[[0,1,19,20,21,23,24,25,26,27,28]], axis=1)
# df_v3 = df_v2[np.isfinite(df_v2['subjective_fun'])]
# df_v3['flag'] = np.where(df_v3['subjective_fun'] > 8, 1, 0)

# # Logistic Regression with train_test_split
# # Assign appropriate independent and dependant variables
# x = df_v3.drop(df_v3[[17,18]], axis=1)
# y = df_v3['flag']

# # Split the data using TTS
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=77)

# # Run the sklearn version of a logistic regression
# lr = LogisticRegression(solver='liblinear')

# # train model using training datasets
# lr_model = lr.fit(X_train, y_train)

# # predict on test dataset using model results
# lr_ypred = lr_model.predict(X_test)

# # Use GridSeachCV with logistic regression
# logreg_parameters = {
#     'penalty':['l1','l2'],
#     'C':np.logspace(-5,1,50),
#     'solver':['liblinear']
# }
# gs = GridSearchCV(lr_model, logreg_parameters, n_jobs=-1, cv=5)

# PREDICTOR = gs.fit(x, y)
# ______________________________
'''INSTEAD of mumbo jumbo above, call in fitted model using pickle:
with open('my_pickled_sample.pkl', 'r') as picklefile:
    the_same_sample = pickle.load(picklefile)

set PREDICTOR = to picklefile'''

# #-------- ROUTES GO HERE -----------#

@app.route('/predict', methods=["GET"])
def predict():
    
    print 'Get Input'
    total_session_len=flask.request.args['total_session_len']
    total_session_amt=flask.request.args['total_session_amt']

    item = [total_session_len, total_session_amt]
    score = PREDICTOR.predict_proba(item)
    results = {'chances of sending (at least 1) booking request': score[0,1], 'chances of no request': score[0,0]}
    return flask.jsonify(results)
     

#---------- CREATING AN API, METHOD 2 ----------------#

# This method takes input via an HTML page
# @app.route('/page')
# def page():
#    with open("input_page.html", 'r') as viz_file:
#        return viz_file.read()

# @app.route('/results', methods=['POST', 'GET'])
# def results():
#     '''Gets prediction using the HTML form'''
#     print "STARTING PREDICTION SESSION"

#     if flask.request.method == 'POST':
#         print "POST RECEIVED"
#         inputs = flask.request.form
#         print "INPUTS FORM"
#         # like_sports = int(inputs['like_sports'][0])
#         # like_tvsports = int(inputs['like_tvsports'][0])
#         # like_exercise = int(inputs['like_exercise'][0])
#         # like_food = int(inputs['like_food'][0])
#         # like_museum = int(inputs['like_museum'][0])
#         # like_art = int(inputs['like_art'][0])
#         # like_hiking = int(inputs['like_hiking'][0])
#         # like_gaming = int(inputs['like_gaming'][0])
#         # like_clubbing = int(inputs['like_clubbing'][0])
#         # like_reading = int(inputs['like_reading'][0])
#         # like_tv = int(inputs['like_tv'][0])
#         # like_theater = int(inputs['like_theater'][0])
#         # like_movies = int(inputs['like_movies'][0])
#         # like_concert = int(inputs['like_concert'][0])
#         # like_music = int(inputs['like_music'][0])
#         # like_shopping = int(inputs['like_shopping'][0])
#         # like_yoga = int(inputs['like_yoga'][0])
#         print "SET VARS"
#         # item = np.array([like_sports, like_tvsports, like_exercise, like_food, like_museum,like_art, like_hiking, like_gaming, like_clubbing, like_reading, like_tv,like_theater, like_movies, like_concert, like_music, like_shopping, like_yoga])
#         print "MAKE ARRAY"
#         score = PREDICTOR.predict_proba(item)
#         print "GOT SCORE"
#         results = {'fun chances': score[0,1], 'non_fun chances': score[0,0]}
#         print 'FINISHED PREDICTING'
#         return flask.jsonify(results)


if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = '4000'

    app.run(HOST,PORT)
    # app.run(debug=True)