from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def feature_importance_analysis(df):
    X = df.drop(columns=["finalgrade", "userid", "courseid"])
    y = df["finalgrade"]

    model = RandomForestRegressor()
    model.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return importance
