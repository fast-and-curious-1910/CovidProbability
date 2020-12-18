from flask import Flask , render_template , redirect
import pickle
import os
import flask
import src.training as training

app = Flask('Covid Infection Probability Detector')
picklefilepath = './data/data.pickle'
picklefile = open(picklefilepath, 'rb')

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

@app.route('/github')
def redirect_to_github():

    return redirect('https://github.com/fast-and-curious-1910/CovidProbability', code=302)

@app.route('/source')
def redirect_to_github2():

    return redirect('https://github.com/fast-and-curious-1910/CovidProbability', code=302)


if __name__ == "__main__":
    app.run(debug=False,port=5000,host='localhost')


