import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('src/data.csv')


def data_split(data, ratio: float):
    np.random.seed(42)
    shuff = np.random.permutation(len(data))
    test_s_s = int(len(data) * ratio)  # Test Set Size
    test_indi = shuff[:test_s_s]  # Test Indices
    train_indi = shuff[test_s_s:]  # Train Indices
    return data.iloc[train_indi], data.iloc[test_indi]

if __name__ == "__main__":

    train, test = data_split(df, 0.2)

    fetlist = ['fever', 'bodyPain', 'age',
            'runnyNose', 'diffBreath']  # Feature List
    
    lablist = ['infectionProb']


    # print(test)

    # print(train)

    x_train = train[fetlist].to_numpy()  # For Features
    x_test = test[fetlist].to_numpy()  # For Features

    y_train = train[lablist].to_numpy().reshape(560,)  # 560 Is the num of rows
    y_test = test[lablist].to_numpy().reshape(139,)  # 139 is num of rows

    clf = LogisticRegression()

    clf.fit(x_train, y_train)

    picklefile = open('src/data.pickle','wb')


    # pickle.load(picklefile)
    pickle.dump(clf,picklefile)

    inputf = [100, 1, 22, 1, 1]
    infProb = clf.predict_proba([inputf])[0][1]
    picklefile.close()
