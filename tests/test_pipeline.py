import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.pipeline import AcademicMoodlePipeline


class TestAcademicMoodlePipeline:
    @pytest.fixture
    def sample_logs(self):
        """Create sample log data for testing."""
        return pd.DataFrame({
            'userid': [1, 1, 1, 2, 2],
            'courseid': [10, 10, 10, 10, 10],
            'timecreated': [1000, 1030, 1100, 2000, 2030],  # Unix timestamps
            'eventname': ['course_viewed', 'resource_viewed', 'quiz_started',
                         'course_viewed', 'discussion_viewed'],
            'component': ['core', 'mod_resource', 'mod_quiz', 'core', 'mod_forum']
        })

    @pytest.fixture
    def sample_grades(self):
        """Create sample grade data for testing."""
        return pd.DataFrame({
            'userid': [1, 2],
            'courseid': [10, 10],
            'itemtype': ['course', 'course'],
            'finalgrade': [85.0, 92.0]
        })

    def test_anonymize_consistent(self):
        """Test that anonymize produces consistent hashes."""
        pipeline = AcademicMoodlePipeline("", "")
        user_id = "test_user"
        hash1 = pipeline._anonymize(user_id)
        hash2 = pipeline._anonymize(user_id)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_anonymize_different_inputs(self):
        """Test that different inputs produce different hashes."""
        pipeline = AcademicMoodlePipeline("", "")
        hash1 = pipeline._anonymize("user1")
        hash2 = pipeline._anonymize("user2")
        assert hash1 != hash2

    @patch('pandas.read_csv')
    @patch('src.pipeline.Path')
    def test_process_basic_functionality(self, mock_path, mock_read_csv, sample_logs, sample_grades):
        """Test basic processing functionality."""
        mock_read_csv.side_effect = [sample_logs, sample_grades]
        mock_path.return_value.exists.return_value = True

        pipeline = AcademicMoodlePipeline("fake_logs.csv", "fake_grades.csv")
        result = pipeline.process()

        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check expected columns are present
        expected_cols = ['userid', 'courseid', 'Num_Sessions', 'Total_Clicks', 'finalgrade']
        for col in expected_cols:
            assert col in result.columns

        # Check data types
        assert result['userid'].dtype == object  # anonymized
        assert result['courseid'].dtype in [int, np.int64]
        assert result['Num_Sessions'].dtype in [int, float, np.int64, np.float64]

    @patch('src.pipeline.Path')
    def test_process_with_empty_data(self, mock_path):
        """Test processing with empty DataFrames."""
        mock_path.return_value.exists.return_value = True
        empty_logs = pd.DataFrame(columns=['userid', 'courseid', 'timecreated', 'eventname'])
        empty_grades = pd.DataFrame(columns=['userid', 'courseid', 'itemtype', 'finalgrade'])

        with patch('pandas.read_csv') as mock_read:
            mock_read.side_effect = [empty_logs, empty_grades]

            pipeline = AcademicMoodlePipeline("", "")
            with pytest.raises(ValueError, match="Logs data is empty"):
                pipeline.process()

    @patch('src.pipeline.Path')
    def test_process_anonymizes_userid(self, mock_path, sample_logs, sample_grades):
        """Test that user IDs are properly anonymized."""
        mock_path.return_value.exists.return_value = True
        with patch('pandas.read_csv') as mock_read:
            mock_read.side_effect = [sample_logs, sample_grades]

            pipeline = AcademicMoodlePipeline("", "")
            result = pipeline.process()

            # Original user IDs should not appear in result
            assert 1 not in result['userid'].values
            assert 2 not in result['userid'].values

            # But should have anonymized versions
            assert len(result['userid'].unique()) > 0

    @patch('src.pipeline.Path')
    def test_process_handles_missing_columns(self, mock_path):
        """Test error handling for missing required columns."""
        mock_path.return_value.exists.return_value = True
        incomplete_logs = pd.DataFrame({'userid': [1], 'timecreated': [1000]})  # missing courseid

        with patch('pandas.read_csv') as mock_read:
            mock_read.side_effect = [incomplete_logs,
                                   pd.DataFrame({'userid': [1], 'courseid': [10], 'itemtype': ['course'], 'finalgrade': [85.0]})]

            pipeline = AcademicMoodlePipeline("", "")

            # Should raise ValueError for missing column
            with pytest.raises(ValueError, match="Required column 'courseid' missing"):
                pipeline.process()