# Moodle Analytics Pipeline

This project analyzes Moodle action logs and grades to understand student engagement and predict academic performance. It includes sessionization, exploratory data analysis (EDA), regression modeling, and feature importance analysis.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/moodle-analytics-pipeline.git

cd moodle-analytics-pipeline

pip install -r requirements.txt

## 🚀 Usage

1. Place raw Moodle logs and grade files in `data/raw/`.
2. Run `pipeline.py` to generate session-level features.
3. Use `eda_plots.py` to visualize engagement and grade patterns.
4. Run `regression.py` to train and compare models.
5. Use `importance.py` to interpret feature impact.

## 📚 References

- Romero & Ventura (2013): Data mining in education.
- Costa et al. (2017): Engagement heterogeneity in LMS.
- Akçapınar et al. (2019): Temporal patterns and performance.

## 📜 License

MIT License
