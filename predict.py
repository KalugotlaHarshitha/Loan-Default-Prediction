
import pandas as pd
import joblib

model=joblib.load('models/model.pkl')
df=pd.read_csv('data/new_data.csv')

pred=model.predict(df)
df['prediction']=pred
df.to_csv('predictions.csv',index=False)
print("Predictions saved to predictions.csv")
