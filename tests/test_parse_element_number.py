import unittest

from data.data_storage import _parse_element_number


class TestParseElementNumber(unittest.TestCase):
    def test_parse_from_python_dict_style_text(self):
        text = "Executing tap with params {'element_number': 12, 'text_input': None}"
        self.assertEqual(_parse_element_number(text), 12)

    def test_parse_missing_value_returns_none(self):
        text = "Executing back with params {}"
        self.assertIsNone(_parse_element_number(text))


if __name__ == "__main__":
    unittest.main()
