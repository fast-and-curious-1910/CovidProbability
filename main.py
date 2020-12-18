from flask import Flask, render_template, redirect, request
import pickle
import os
import flask
import src.training as training

app = Flask('Covid Infection Probability Detector')
picklefilepath = './data/data.pickle'
picklefile = open(picklefilepath, 'rb')

clf = pickle.load(file=picklefile)

picklefile.close()


@app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'POST':

        mainDict1 = request.form
        fever = int(mainDict1['fever'])
        age = int(mainDict1['age'])
        pain = int(mainDict1['pain'])
        runnyNose = int(mainDict1['runnyNose'])
        diffBreath = int(mainDict1['diffBreath'])

        inputf = [fever, pain, age, runnyNose, diffBreath]
        # inputf = "i like cheese"
        infProb = clf.predict_proba([inputf])[0][1]
        return render_template('showProb.html', inf=round(infProb*100))

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
    app.run(debug=False, port=5000, host='localhost')
