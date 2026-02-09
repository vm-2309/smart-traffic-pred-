import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path

print("Starting Smart Traffic Model Training...")

# small fake traffic dataset
data = pd.DataFrame({
    "hour": [1,2,3,4,5,6,7,8,9,10],
    "vehicles": [10,15,20,30,50,80,120,100,70,40]
})

X = data[["hour"]]
y = data["vehicles"]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[11]])[0]

Path("reports").mkdir(exist_ok=True)
(Path("reports") / "prediction.txt").write_text(
    f"Predicted vehicles at hour 11 = {prediction:.2f}"
)

print("Training finished successfully!")
