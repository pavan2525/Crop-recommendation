from flask import Flask, request, render_template
# import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))
df = pd.read_csv('Crop_recommendation.csv')
encode_soil = LabelEncoder()

#fitting the label encoder
df.soiltype = encode_soil.fit_transform(df.soiltype)

#creating the DataFrame
soiltype = pd.DataFrame(zip(encode_soil.classes_,encode_soil.transform(encode_soil.classes_)),columns=['Original','Encoded'])
soiltype = soiltype.set_index('Original')

encode_weather = LabelEncoder()

#fitting the label encoder
df.weather = encode_weather.fit_transform(df.weather)

#creating the DataFrame
weather = pd.DataFrame(zip(encode_weather.classes_,encode_weather.transform(encode_weather.classes_)),columns=['Original','Encoded'])
weather = weather.set_index('Original')

encode_location = LabelEncoder()

#fitting the label encoder
df.location = encode_location.fit_transform(df.location)

#creating the DataFrame
location = pd.DataFrame(zip(encode_location.classes_,encode_location.transform(encode_location.classes_)),columns=['Original','Encoded'])
location = location.set_index('Original')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
scaler = StandardScaler()
X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,shuffle=True,stratify=y)
model=RandomForestClassifier(n_estimators=1000,random_state=123)
model.fit(X_train,y_train)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    # y_pred=model.predict(X_test)
    # score=accuracy_score(y_test,y_pred)
    # print("Accuracy:", score)
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array([int_features])
    print(final_features)
    # scaler = StandardScaler()
    data = scaler.transform(final_features)
    prediction = model.predict_proba(data)
    print(prediction)
    # output = prediction[0]
    # print(output)

    # return render_template('index.html', prediction_text='Ideal Crop to grow is {}'.format(output))
    top_5_crops_indices = np.argsort(prediction[0])[::-1][:5]
    print(top_5_crops_indices)
    top_5_crops = [model.classes_[idx] for idx in top_5_crops_indices]
    print(top_5_crops)
    
    return render_template('index.html', prediction_text='Top 5 ideal crops to grow are: {}'.format(', '.join(top_5_crops)))

if __name__ == "__main__":
    app.run(debug=True)
