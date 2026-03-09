import pandas as pd
import hashlib
from pathlib import Path
import yaml


class AcademicMoodlePipeline:
    def __init__(self, log_path, grade_path, config_path="config.yaml", salt=None):
        self.log_path = log_path
        self.grade_path = grade_path

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.salt = salt or self.config['anonymization']['salt']
        self.inactivity_threshold = self.config['data']['inactivity_threshold_minutes']
        self.max_gap = self.config['data']['max_gap_minutes']

    def _validate_data(self, logs, grades):
        """Validate input data for required columns and basic integrity."""
        required_log_cols = ['userid', 'courseid', 'timecreated', 'eventname']
        required_grade_cols = ['userid', 'courseid', 'itemtype', 'finalgrade']

        # Check required columns
        for col in required_log_cols:
            if col not in logs.columns:
                raise ValueError(f"Required column '{col}' missing from logs data")

        for col in required_grade_cols:
            if col not in grades.columns:
                raise ValueError(f"Required column '{col}' missing from grades data")

        # Check for empty data
        if logs.empty:
            raise ValueError("Logs data is empty")
        if grades.empty:
            raise ValueError("Grades data is empty")

        # Check data types
        if not pd.api.types.is_numeric_dtype(logs['timecreated']):
            raise ValueError("timecreated column must be numeric (Unix timestamp)")

        if not pd.api.types.is_numeric_dtype(grades['finalgrade']):
            raise ValueError("finalgrade column must be numeric")

        # Check for course grades
        course_grades = grades[grades['itemtype'] == 'course']
        if course_grades.empty:
            raise ValueError("No course-level grades found in grades data")

        print(f"✓ Data validation passed: {len(logs)} log entries, {len(grades)} grade entries")

    def _anonymize(self, user_id):
        return hashlib.sha256(f"{user_id}{self.salt}".encode()).hexdigest()

    def process(self):
        # Check file existence
        if not Path(self.log_path).exists():
            raise FileNotFoundError(f"Logs file not found: {self.log_path}")
        if not Path(self.grade_path).exists():
            raise FileNotFoundError(f"Grades file not found: {self.grade_path}")

        logs = pd.read_csv(self.log_path, index_col=0)
        grades = pd.read_csv(self.grade_path, index_col=0)

        # Validate data
        self._validate_data(logs, grades)

        logs["time"] = pd.to_datetime(logs["timecreated"], unit="s")
        logs = logs.sort_values(["userid", "courseid", "time"])

        logs["gap_min"] = logs.groupby(["userid", "courseid"])["time"].diff().dt.total_seconds().div(60)
        logs["gap_min"] = logs["gap_min"].clip(upper=self.max_gap)

        logs["new_session"] = (logs["gap_min"] > self.inactivity_threshold) | (logs["gap_min"].isna())
        logs["session_id"] = logs.groupby(["userid", "courseid"])["new_session"].cumsum()

        def compute_features(data):
            session_bounds = data.groupby("session_id")["time"].agg(["min", "max"])
            session_duration = (session_bounds["max"] - session_bounds["min"]).dt.total_seconds().div(60)
            session_starts = data.loc[data["new_session"], "time"]
            deltas = session_starts.diff().dt.total_seconds().div(60)

            return pd.Series({
                "Num_Sessions": data["session_id"].nunique(),
                "Total_Clicks": len(data),
                "Total_Time_Online": session_duration.sum(),
                "Avg_Time_Per_Session": session_duration.mean(),
                "Session_Start_Hour_STD": session_starts.dt.hour.std() if len(session_starts) > 1 else 0,
                "Session_Interval_CV": (deltas.std() / deltas.mean() if deltas.mean() and len(deltas) > 1 else 0),
                "Max_Inactivity_Period": data["gap_min"].max(),
                "Course_Views": data["eventname"].str.contains("course_viewed").sum(),
                "Module_Views": data["eventname"].str.contains("course_module_viewed").sum(),
                "Content_Views": data["eventname"].str.contains("resource|book").sum(),
                "Forum_Posts": data["eventname"].str.contains("post_created").sum(),
                "Forum_Views": data["eventname"].str.contains("discussion_viewed").sum(),
                "Messages_Sent": data["eventname"].str.contains("message_sent").sum(),
                "Quiz_Started": data["eventname"].str.contains("attempt_started").sum(),
                "Quiz_Submitted": data["eventname"].str.contains("attempt_submitted").sum(),
                "Quiz_Viewed": data["eventname"].str.contains("attempt_viewed").sum(),
                "Quiz_Reviewed": data["eventname"].str.contains("attempt_reviewed").sum(),
                "Assignments_Submitted": data["eventname"].str.contains("assessable_submitted").sum(),
                "Assignment_Views": data["eventname"].str.contains("submission_viewed").sum(),
                "Files_Downloaded": data["eventname"].str.contains("files_downloaded").sum(),
                "Files_Uploaded": data["eventname"].str.contains("assessable_uploaded").sum(),
                "Meeting_Joined": data["eventname"].str.contains("meeting_joined").sum(),
            })

        features = logs.groupby(["userid", "courseid"]).apply(compute_features, include_groups=False).reset_index()

        final_grades = grades[grades["itemtype"] == "course"].groupby(["userid", "courseid"])["finalgrade"].max().reset_index()

        final_df = features.merge(final_grades, on=["userid", "courseid"], how="inner")
        final_df["userid"] = final_df["userid"].apply(self._anonymize)

        return final_df.fillna(0)
