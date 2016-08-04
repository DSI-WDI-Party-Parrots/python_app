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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# from sklearn.cross_validation import train_test_split, cross_val_score
# from sklearn.metrics import roc_curve
# from sklearn.neighbors import KNeighborsClassifier


userdata = pd.read_csv('user_airbnbdata.csv')

requestmodel,searchmodel,messagemodel = LogisticRegressionCV(),LogisticRegressionCV(),LogisticRegressionCV()
X = userdata[['SessionLength','NumSessions']]
yrequest = userdata.requestsentbinary
ymessage = userdata.messagesentbinary
ysearch = userdata.searchbinary

requestmodel = requestmodel.fit(X,yrequest)
messagemodel = messagemodel.fit(X,ymessage)
searchmodel = searchmodel.fit(X,ysearch)


# ______________________________
'''INSTEAD of mumbo jumbo above, call in fitted model using pickle:
with open('my_pickled_sample.pkl', 'r') as picklefile:
    the_same_sample = pickle.load(picklefile)

set PREDICTOR = to picklefile'''

# #-------- ROUTES GO HERE -----------#

# @app.route('/predict', methods=["GET"])
# def predict():
    
#     print 'Get Input'
#     SessionLength=flask.request.args['SessionLength']
#     NumSessions=flask.request.args['NumSessions']

#     item = [SessionLength, NumSessions]
#     score = requestmodel.predict_proba(item)
#     results = {'chances of sending (at least 1) booking request': score[0,1], 'chances of no request': score[0,0]}
#     return flask.jsonify(results)


# ---------- CREATING AN API, METHOD 2 ----------------#

# This method takes input via an HTML page
@app.route('/page')
def page():
   with open("input_page.html", 'r') as viz_file:
       return viz_file.read()

@app.route('/results', methods=['POST', 'GET'])
def results():
    '''Gets prediction using the HTML form'''
    print "STARTING PREDICTION SESSION"

    if flask.request.method == 'POST':
        print "POST RECEIVED"
        inputs = flask.request.form
        print "INPUTS FORM"

        SessionLength = int(inputs['SessionLength'][0])
        NumSessions = int(inputs['NumSessions'][0])
        # like_exercise = int(inputs['like_exercise'][0])
        # like_food = int(inputs['like_food'][0])
        # like_museum = int(inputs['like_museum'][0])
        # like_art = int(inputs['like_art'][0])
        # like_hiking = int(inputs['like_hiking'][0])
        # like_gaming = int(inputs['like_gaming'][0])
        # like_clubbing = int(inputs['like_clubbing'][0])
        # like_reading = int(inputs['like_reading'][0])
        # like_tv = int(inputs['like_tv'][0])
        # like_theater = int(inputs['like_theater'][0])
        # like_movies = int(inputs['like_movies'][0])
        # like_concert = int(inputs['like_concert'][0])
        # like_music = int(inputs['like_music'][0])
        # like_shopping = int(inputs['like_shopping'][0])
        # like_yoga = int(inputs['like_yoga'][0])
        print "SET VARS"
        item = np.array([SessionLength, NumSessions])
        print "MAKE ARRAY"
        score = requestmodel.predict_proba(item)
        print "GOT SCORE"
        results = {'chances of sending (at least 1) booking request': score[0,1], 'chances of no request': score[0,0]}
        print 'FINISHED PREDICTING'
        return flask.jsonify(results)


if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = '4000'

    app.run(HOST,PORT)
    # app.run(debug=True)