# Moodle Analytics Pipeline

This project analyzes Moodle action logs and grades to understand student engagement and predict academic performance. It includes sessionization, exploratory data analysis (EDA), regression modeling, and feature importance analysis.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/moodle-analytics-pipeline.git
cd moodle-analytics-pipeline
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Or use the Makefile:
```bash
make install
```

### 3. Initial Setup
```bash
make setup
```
This creates the necessary directories and provides setup instructions.

### 4. Place Data Files
Put your Moodle data files in `data/raw/`:
- `logs.csv`: Moodle log data with columns: userid, courseid, timecreated, eventname, etc.
- `grades.csv`: Gradebook data with columns: userid, courseid, itemtype, finalgrade, etc.

### 5. Run the Pipeline
```bash
make run-pipeline
```

Or manually:
```python
from src.pipeline import AcademicMoodlePipeline

# Process data
pipeline = AcademicMoodlePipeline("data/raw/logs.csv", "data/raw/grades.csv")
df = pipeline.process()

# Run analysis
from src.eda_plots import run_eda_plots
run_eda_plots(df)

from src.regression import run_regression_models
run_regression_models(df)
```

### 6. Run Analysis
```bash
make run-eda      # Exploratory data analysis
make run-models   # Model training and evaluation
```

### 5. Check Results
- Processed features: `data/processed/session_features.csv`
- Model results: `report/model_results.csv`
- SHAP analysis: `report/shap/`

---

## 📖 Usage Examples

### Basic Data Processing
```python
from src.pipeline import AcademicMoodlePipeline

pipeline = AcademicMoodlePipeline(
    log_path="data/raw/logs.csv",
    grade_path="data/raw/grades.csv"
)
features_df = pipeline.process()
print(f"Processed {len(features_df)} student-course combinations")
```

### Exploratory Data Analysis
```python
from src.eda_plots import run_eda_plots
import pandas as pd

df = pd.read_csv("data/processed/session_features.csv")
run_eda_plots(df)  # Generates plots in notebook
```

### Model Training and Evaluation
```python
from src.regression import run_regression_models
import pandas as pd

df = pd.read_csv("data/processed/session_features.csv")
results = run_regression_models(df)
print(results.head())
```

### Feature Importance
```python
from src.importance import feature_importance_analysis
import pandas as pd

df = pd.read_csv("data/processed/session_features.csv")
importance_df = feature_importance_analysis(df)
print(importance_df)
```

---

## 🔧 Configuration

The pipeline uses a `config.yaml` file for parameters:

```yaml
data:
  inactivity_threshold_minutes: 30
  max_gap_minutes: 1440

model:
  test_size: 0.2
  cv_folds: 5
  random_forest:
    n_estimators: [100, 200, 300]
    max_depth: [null, 10, 20]
```

Modify these values to customize the analysis.

---

## �️ Development

### Makefile Commands
The project includes a Makefile for common tasks:

```bash
make help        # Show all available commands
make install     # Install dependencies
make test        # Run unit tests
make clean       # Clean up generated files
make setup       # Initial project setup
make run-pipeline # Run data processing
make run-eda     # Run exploratory analysis
make run-models  # Run model training
```

### Testing
Run unit tests:
```bash
make test
# or
pytest tests/ -v
```

---

## �📊 Expected Outputs

### Processed Features (23 features per student-course)
- **Session features**: Num_Sessions, Total_Clicks, Total_Time_Online, Avg_Time_Per_Session
- **Temporal features**: Session_Start_Hour_STD, Session_Interval_CV, Max_Inactivity_Period
- **Activity features**: Course_Views, Module_Views, Content_Views, Forum_Posts, etc.
- **Target**: finalgrade (0-100)

### Model Performance (Typical Results)
| Model | CV R² Mean | CV R² Std | Test R² |
|-------|------------|-----------|---------|
| Random Forest (Optimized) | 0.35 | 0.08 | 0.38 |
| Gradient Boosting | 0.32 | 0.09 | 0.35 |
| XGBoost | 0.31 | 0.08 | 0.34 |

### SHAP Analysis
- Global feature importance plot
- Per-student top 5 influential features
- LLM-ready dataset for feedback generation

---

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

---

## 📚 References

- Romero & Ventura (2013): Data mining in education.
- Costa et al. (2017): Engagement heterogeneity in LMS.
- Akçapınar et al. (2019): Temporal patterns and performance.

---

## 📜 License

MIT License
