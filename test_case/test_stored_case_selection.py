import unittest

from ui import select_stored_cases, stored_case_choices


class StoredCaseSelectionTests(unittest.TestCase):
    def setUp(self):
        self.actions = [
            {"action_id": "case-a", "source_task": "Open Gallery"},
            {"action_id": "case-b", "name": "Start timer"},
            {"name": "Missing identifier"},
            {"action_id": "case-c"},
        ]

    def test_choices_have_labels_and_skip_unusable_records(self):
        self.assertEqual(
            stored_case_choices(self.actions),
            [
                ("Open Gallery", "case-a"),
                ("Start timer", "case-b"),
                ("case-c", "case-c"),
            ],
        )

    def test_selected_cases_follow_explicit_selection_order(self):
        selected = select_stored_cases(
            self.actions,
            ["case-b", "missing", "case-a", "case-b"],
        )
        self.assertEqual([case["action_id"] for case in selected], ["case-b", "case-a"])

    def test_empty_selection_does_not_select_every_case(self):
        self.assertEqual(select_stored_cases(self.actions, []), [])


if __name__ == "__main__":
    unittest.main()
