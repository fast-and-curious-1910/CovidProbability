from flask import Flask , render_template
import pickle

import flask
import src.training as training

app = Flask('Covid Infection Probability Detector')

picklefile = open('src/data.pickle', 'rb')

clf = pickle.load(file=picklefile)

picklefile.close()
@app.route('/')
def main():
    inputf = [100, 1, 22, 1, 1]
    infProb = clf.predict_proba([inputf])[0][1]
    return render_template('index.html')


@app.errorhandler(404)
def err(error):
    return flask.Response(status=404)




if __name__ == "__main__":
    app.run(debug=False,port=5000,host='0.0.0.0')


