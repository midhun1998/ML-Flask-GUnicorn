import sklearn
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import pickle
import os
iris = sklearn.datasets.load_iris()
training, testing, training_labels, testing_labels = sklearn.model_selection.train_test_split(iris.data, iris.target, train_size=0.80)

rfcl = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rfcl.fit(training, training_labels)
os.chdir('/mnt/d/Dev/Projects/Stock Prediction Flask and Gunicorn/iris-flask-app/predict_app/')
filename = 'iris_model.pkl'
pickle.dump(rfcl, open(filename, 'wb'))
load_model = pickle.load(open(filename, 'rb'))
result = load_model.score(testing, testing_labels)
print(result)