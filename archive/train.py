import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

data = {
    'pace':      [99, 90, 80, 70, 60, 50, 40, 30],
    'shooting':  [99, 90, 80, 70, 60, 50, 40, 30],
    'passing':   [99, 90, 80, 70, 60, 50, 40, 30],
    'dribbling': [99, 90, 80, 70, 60, 50, 40, 30],
    'physic':    [99, 90, 80, 70, 60, 50, 40, 30],
    'overall':   [99, 90, 80, 70, 60, 50, 40, 30] 
}

df = pd.DataFrame(data)

features = ['pace', 'shooting', 'passing', 'dribbling', 'physic']
X = df[features]
y = df['overall']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'modelo_fifa.pkl')

print("✅ Modelo RECALIBRADO! Agora ele reconhece jogadores ruins.")