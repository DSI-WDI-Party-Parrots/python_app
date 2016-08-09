import flask
app = flask.Flask(__name__)

#---------this app route is just for testing purposes....--------#
@app.route("/hello")
def hello():
    print "Hello World!!"
    return "Hello World!"

#-------- MODEL GOES HERE -----------#
# import pickle

# with open('pickled_model.pkl', 'r') as picklefile:
#     requestmodel = pickle.load(picklefile)


# import numpy as np
# import pandas as pd
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
# # from sklearn.cross_validation import train_test_split, cross_val_score
# # from sklearn.metrics import roc_curve
# # from sklearn.neighbors import KNeighborsClassifier


# userdata = pd.read_csv('user_airbnbdata.csv')

# requestmodel,searchmodel,messagemodel = LogisticRegressionCV(),LogisticRegressionCV(),LogisticRegressionCV()
# X = userdata[['SessionLength','NumSessions']]
# yrequest = userdata.requestsentbinary
# ymessage = userdata.messagesentbinary
# ysearch = userdata.searchbinary

# requestmodel = requestmodel.fit(X,yrequest)
# messagemodel = messagemodel.fit(X,ymessage)
# searchmodel = searchmodel.fit(X,ysearch)


# #-------- ROUTES GO HERE -----------#

#localhost:4000/predict?SessionLength=7000&NumSessions=2

@app.route('/predict', methods=["GET"])
def predict():
    
    from flask import request

    SessionLength=int(request.args['SessionLength'])
    NumSessions=int(request.args['NumSessions'])

    item = [SessionLength, NumSessions]
    score_request = requestmodel.predict_proba(item)
    # score_message = messagemodel.predict_proba(item)
    # score_search = searchmodel.predict_proba(item)
    results_request = [{'chances_booking_request': score_request[0,1], 'no_request_chance': score_request[0,0]}]
    # results_message = {'chances of sending (at least 1) message': score_message[0,1], 'chances of no message': score_message[0,0]}
    # results_search = {'chances of doing (at least 1) search': score_search[0,1], 'chances of no search': score_search[0,0]}


    return flask.jsonify(results_request)
    # return flask.jsonify(results_message)
    # return flask.jsonify(results_search)


# ---------- CREATING AN API, METHOD 2 ----------------#

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

#         SessionLength = int(inputs['SessionLength'][0])
#         NumSessions = int(inputs['NumSessions'][0])
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
        
#         # item = np.array([SessionLength, NumSessions])
#         # score = requestmodel.predict_proba(item)
#         # results = {'chances of sending (at least 1) booking request': score[0,1], 'chances of no request': score[0,0]}
        
#         item = [SessionLength, NumSessions]
#         score_request = requestmodel.predict_proba(item)
#         score_message = messagemodel.predict_proba(item)
#         score_search = searchmodel.predict_proba(item)
#         results_request = {'chances of sending (at least 1) booking request': score_request[0,1], 'chances of no request': score_request[0,0]}
#         results_message = {'chances of sending (at least 1) message': score_message[0,1], 'chances of no message': score_message[0,0]}
#         results_search = {'chances of doing (at least 1) search': score_search[0,1], 'chances of no search': score_search[0,0]}

#         return flask.jsonify(results_request)
#         return flask.jsonify(results_message)
#         return flask.jsonify(results_search)

        # return flask.jsonify(results)


if __name__ == '__main__':
    '''Connects to the server'''

    HOST = '127.0.0.1'
    PORT = '4000'

    # app.run(HOST,PORT)
    app.run(debug=True)