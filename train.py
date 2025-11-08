
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

df=pd.read_csv('data/loan_data.csv')
X=df[['income','loan_amount','credit_score']]
y=df['default']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

model=LogisticRegression()
model.fit(X_train,y_train)

joblib.dump(model,'models/model.pkl')
print("Model trained and saved to models/model.pkl")
