import pandas as pd
import hashlib

class AcademicMoodlePipeline:
    def __init__(self, log_path, grade_path, salt="PhD_Project_2026"):
        self.log_path = log_path
        self.grade_path = grade_path
        self.salt = salt

    def _anonymize(self, user_id):
        return hashlib.sha256(f"{user_id}{self.salt}".encode()).hexdigest()

    def process(self):
        logs = pd.read_csv(self.log_path, index_col=0)
        grades = pd.read_csv(self.grade_path, index_col=0)

        logs["time"] = pd.to_datetime(logs["timecreated"], unit="s")
        logs = logs.sort_values(["userid", "courseid", "time"])

        logs["gap_min"] = logs.groupby(["userid", "courseid"])["time"].diff().dt.total_seconds().div(60)
        logs["gap_min"] = logs["gap_min"].clip(upper=1440)

        logs["new_session"] = (logs["gap_min"] > 30) | (logs["gap_min"].isna())
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
