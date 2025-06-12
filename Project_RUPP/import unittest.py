import unittest
from unittest.mock import patch, MagicMock, mock_open
import builtins
from new_version_face_realtime import recognize_and_send
import new_version_face_realtime
import new_version_face_realtime

# File: tests/test_new_version_face_realtime.py


# Absolute import for the function to test

class TestRecognizeAndSend(unittest.TestCase):
    @patch("new_version_face_realtime.cv2.VideoCapture")
    @patch("new_version_face_realtime.cv2.imshow")
    @patch("new_version_face_realtime.cv2.waitKey")
    @patch("new_version_face_realtime.cv2.destroyAllWindows")
    @patch("new_version_face_realtime.face_recognition.face_locations")
    @patch("new_version_face_realtime.face_recognition.face_encodings")
    @patch("new_version_face_realtime.face_recognition.face_distance")
    @patch("new_version_face_realtime.requests.post")
    @patch("new_version_face_realtime.load_submission_cache")
    @patch("new_version_face_realtime.save_submission_cache")
    def test_known_face_submission(
        self, mock_save_cache, mock_load_cache, mock_post, mock_face_distance,
        mock_face_encodings, mock_face_locations, mock_destroy, mock_waitKey,
        mock_imshow, mock_VideoCapture
    ):
        # Setup mocks
        mock_cap = MagicMock()
        mock_VideoCapture.return_value = mock_cap
        mock_cap.read.side_effect = [
            (True, "frame"),  # First iteration
            (False, None)     # End loop
        ]
        mock_cap.get.side_effect = lambda x: 400  # frame width/height

        mock_face_locations.return_value = [(0, 1, 2, 3)]
        mock_face_encodings.return_value = [[0.1, 0.2, 0.3]]
        mock_face_distance.return_value = [0.3]  # Below threshold

        # Simulate known_encodings and known_names in module
        new_version_face_realtime.known_encodings = [[0.1, 0.2, 0.3]]
        new_version_face_realtime.known_names = ["Alice"]

        mock_load_cache.return_value = {}
        mock_post.return_value.text = "Success"
        mock_waitKey.side_effect = [ord('q')]  # Exit after one loop

        recognize_and_send()

        # Check that post was called (data sent)
        self.assertTrue(mock_post.called)
        # Check that cache was saved
        self.assertTrue(mock_save_cache.called)

    @patch("new_version_face_realtime.cv2.VideoCapture")
    @patch("new_version_face_realtime.cv2.imshow")
    @patch("new_version_face_realtime.cv2.waitKey")
    @patch("new_version_face_realtime.cv2.destroyAllWindows")
    @patch("new_version_face_realtime.face_recognition.face_locations")
    @patch("new_version_face_realtime.face_recognition.face_encodings")
    @patch("new_version_face_realtime.face_recognition.face_distance")
    @patch("new_version_face_realtime.requests.post")
    @patch("new_version_face_realtime.load_submission_cache")
    @patch("new_version_face_realtime.save_submission_cache")
    def test_unknown_face_no_submission(
        self, mock_save_cache, mock_load_cache, mock_post, mock_face_distance,
        mock_face_encodings, mock_face_locations, mock_destroy, mock_waitKey,
        mock_imshow, mock_VideoCapture
    ):
        # Setup mocks
        mock_cap = MagicMock()
        mock_VideoCapture.return_value = mock_cap
        mock_cap.read.side_effect = [
            (True, "frame"),  # First iteration
            (False, None)     # End loop
        ]
        mock_cap.get.side_effect = lambda x: 400  # frame width/height

        mock_face_locations.return_value = [(0, 1, 2, 3)]
        mock_face_encodings.return_value = [[0.1, 0.2, 0.3]]
        mock_face_distance.return_value = [0.9]  # Above threshold

        new_version_face_realtime.known_encodings = [[0.1, 0.2, 0.3]]
        new_version_face_realtime.known_names = ["Alice"]

        mock_load_cache.return_value = {}
        mock_post.return_value.text = "Success"
        mock_waitKey.side_effect = [ord('q')]  # Exit after one loop

        recognize_and_send()

        # Should not call post for unknown face
        self.assertFalse(mock_post.called)
        # Should not save cache for unknown face
        self.assertFalse(mock_save_cache.called)

if __name__ == "__main__":
    unittest.main()