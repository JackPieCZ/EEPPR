import os
import unittest
from loader import get_sequence_file_path

SEQ_NAMES = [
    "highcontrastline",
    "velcro_front",
    "velcro_side",
    "highcontrastdot",
    "handspinner",
    "spider",
    "led",
    "screen",
    "speaker",
    "motor",
    "chain_side",
    "chain_top"
]


class TestLoader(unittest.TestCase):
    def test_get_sequence_file_path(self):
        """
        Test getting the file path for a sequence.
        """
        for seq_name in SEQ_NAMES:
            file_path = get_sequence_file_path(seq_name)
            self.assertIsNotNone(
                file_path, msg=f"{seq_name} file path is None.")
            self.assertTrue(os.path.isfile(file_path),
                            msg=f"{seq_name} file does not exist.")

    def test_remove_file(self):
        """
        Test removing a file after it has been downloaded.
        """
        files_to_remove = ['highcontrastdot', 'handspinner', 'motor']
        for file_name in files_to_remove:
            file_path = get_sequence_file_path(file_name)
            os.remove(file_path)
            self.assertFalse(os.path.isfile(file_path),
                             msg=f"{file_name} file still exists after removal.")


if __name__ == '__main__':
    unittest.main()
