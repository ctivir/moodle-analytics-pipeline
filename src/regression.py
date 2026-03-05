import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
from xgboost import XGBRegressor

def run_regression_models(df, reports_dir="../report"):
    """
    Run regression models to predict 'finalgrade' from Moodle features.
    Exports evaluation metrics, SHAP explainability results, and
    per-student top SHAP features for LLM feedback.
    """
    
    # Ensure reports_dir is a Path object
    reports_dir = Path(reports_dir)

    # 1. Prepare data
    X = df.drop(columns=["finalgrade", "userid", "courseid"])
    y = df["finalgrade"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Baseline and ensemble models
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "MSE": mean_squared_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        }

    # 3. Hyperparameter optimization (Random Forest)
    param_grid_rf = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    }
    grid_rf = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid_rf,
        cv=3,
        scoring="r2",
        n_jobs=-1
    )
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    print("Best RF parameters:", grid_rf.best_params_)

    y_pred_rf = best_rf.predict(X_test)
    results["Random Forest (Optimized)"] = {
        "MSE": mean_squared_error(y_test, y_pred_rf),
        "R2": r2_score(y_test, y_pred_rf)
    }

    # 4. Export metrics
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results).T
    results_df.to_csv(reports_dir / "model_results.csv")
    print(results_df)

    # 5. SHAP explainability (using optimized RF)
    shap_dir = reports_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(best_rf)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.savefig(shap_dir / "shap_global_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_test.columns
    ).sort_values(ascending=False)
    mean_abs_shap.to_csv(shap_dir / "shap_global_mean_abs.csv")

    # 6. Export per-student SHAP dataset (top 5 features per student)
    shap_values_df = pd.DataFrame(shap_values, columns=X_test.columns)
    shap_values_df["userid"] = df.loc[X_test.index, "userid"].values
    shap_values_df["finalgrade"] = y_test.values

    llm_ready_rows = []
    for _, row in shap_values_df.iterrows():
        student_features = row.drop(["userid", "finalgrade"])
        top_features = student_features.abs().sort_values(ascending=False).head(5).index
        top_values = student_features[top_features]
        llm_ready_rows.append({
            "userid": row["userid"],
            "finalgrade": row["finalgrade"],
            "top_features": ", ".join(top_features),
            "top_values": ", ".join([f"{v:.3f}" for v in top_values])
        })

    llm_ready_df = pd.DataFrame(llm_ready_rows)
    llm_ready_df.to_csv(shap_dir / "llm_ready_dataset.csv", index=False)
    print("LLM-ready dataset exported to:", shap_dir / "llm_ready_dataset.csv")

    return results_df
