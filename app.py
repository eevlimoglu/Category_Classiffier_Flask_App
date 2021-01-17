from flask import Flask,render_template, request, redirect, url_for
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from flask_sqlalchemy import SQLAlchemy
from sklearn.model_selection import GridSearchCV

import sklearn
import pickle

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////Users/eevli/Desktop/Category_Review_Classifier/amazon.db'
db = SQLAlchemy(app)

@app.route('/')
def home():

    feeds = amazon.query.all()
    return render_template('index.html', feeds=feeds)


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    feed = request.form.get('feed')
    mail = request.form.get("mail")
           
    with open('./models/category_classifier.pkl', 'rb') as f:
        knn = pickle.load(f)
    with open('./models/category_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('./models/sentiment_classifier.pkl', 'rb') as f:
        clf = pickle.load(f)

    arr = [feed]
    new_arr = vectorizer.transform(arr)
    predict = knn.predict(new_arr)

    newfeed = amazon(mail=mail, feedback=feed, pred=predict)
    db.session.add(newfeed)
    db.session.commit()

    return render_template("result.html", data=predict)

class amazon(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    mail = db.Column(db.Text)
    feedback = db.Column(db.Text)
    want = db.Column(db.Integer)
    pred = db.Column(db.Text)
        
if __name__ == "__main__":
    app.run(debug=True)