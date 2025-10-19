import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Sample diabetes dataset
data = pd.DataFrame({
    'Age':[25, 30, 45, 50, 35],
    'Glucose':[120, 140, 180, 160, 130],
    'BloodPressure':[80, 90, 100, 95, 85],
    'Outcome':[0, 0, 1, 1, 0]
})

X = data[['Age','Glucose','BloodPressure']]
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('models/diabetes_model.pkl','wb'))
print("Model trained and saved.")
