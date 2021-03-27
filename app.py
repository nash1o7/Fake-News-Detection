from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

app = Flask(__name__)
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
# Importing fake news dataset
fake=pd.read_csv('politifact_fake.csv')
fake["Remarks"]="fake"

# Importing real news dataset
real=pd.read_csv('politifact_real.csv')
real["Remarks"]="real"

#joining the 2 dataframes
merged=pd.concat([fake,real],join="inner")
dataframe =merged.copy()
dataframe.head()
x = dataframe['title']
y = dataframe['Remarks']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(news):
    tfid_x_train = tfvect.fit_transform(x_train.apply(lambda x: np.str_(x)))
    tfid_x_test = tfvect.transform(x_test.apply(lambda x: np.str_(x)))
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)