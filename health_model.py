import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Reading the data
df = pd.read_csv("diabetes_prediction_dataset.csv")
print(df.head())

df['gender']=df['gender'].map({"Male":1,"Other":0,"Female":2})
df['gender'].groupby(df.gender).count()

#df['heart_disease']=df['heart_disease'].map({0:'No',1:"Yes"})

df['smoking_history'] = df['smoking_history'].fillna(-1).map({'never':0, 'No Info':1, 'current':2, 'former':3,'ever':4,'not current':5})
df.head()
df['smoking_history'].groupby(df.smoking_history).count()
y = df['diabetes']
df.drop(columns='diabetes', inplace=True)
#x = df['gender', 'age', 'hypertension', 'heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']
x=df

# Training the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))