from datetime import datetime

import pytest

from smart_task_manager import TaskIntelligence

ai = TaskIntelligence()
MONDAY = datetime(2026, 10, 19, 9, 0)  # a Monday


def test_deadline_phrases():
    assert ai.parse_deadline_days("finish in 3 days", MONDAY) == 3
    assert ai.parse_deadline_days("reply within 12 hours", MONDAY) == 0.5
    assert ai.parse_deadline_days("send it tomorrow", MONDAY) == 1
    assert ai.parse_deadline_days("call the client today", MONDAY) == 0.5
    assert ai.parse_deadline_days("no date here", MONDAY) is None


def test_weekday_deadline_counts_forward():
    assert ai.parse_deadline_days("report due friday", MONDAY) == 4
    # Same weekday means next week, not today.
    assert ai.parse_deadline_days("report by monday", MONDAY) == 7


def test_urgent_task_scores_higher_than_calm_one():
    urgent = ai.priority_score("Urgent critical fix", "needed asap, due tomorrow")
    calm = ai.priority_score("Tidy bookshelf")
    assert urgent > calm


@pytest.mark.parametrize("title", ["", "x", "urgent " * 50])
def test_priority_stays_between_0_and_10(title):
    assert 0 <= ai.priority_score(title, "due tomorrow, asap") <= 10


def test_duration_keywords_match_whole_words_only():
    # "shortcut" contains "short" but is a different word.
    assert ai.base_duration("Learn a keyboard shortcut") == 2.0
    assert ai.base_duration("Short task") == 1.0


def test_complex_words_raise_and_simple_words_lower_the_estimate():
    assert ai.base_duration("Design the schema") > ai.base_duration("Update the schema")


def test_calibration_needs_enough_history():
    assert ai.calibration_factor([]) == 1.0
    assert ai.calibration_factor([(2, 4), (2, 4)]) == 1.0


def test_calibration_learns_from_history():
    history = [(2, 4), (2, 3), (1, 2)]  # tasks ran about twice as long as estimated
    assert ai.calibration_factor(history) == pytest.approx(1.833, abs=0.001)


def test_calibration_is_clamped():
    assert ai.calibration_factor([(1, 50)] * 5) == 2.0
    assert ai.calibration_factor([(10, 1)] * 5) == 0.5


def test_calibration_ignores_missing_durations():
    assert ai.calibration_factor([(2, 0), (2, None), (None, 3), (2, 4)]) == 1.0


def test_category_suggestion():
    assert ai.suggest_category("Prepare the client presentation") == "work"
    assert ai.suggest_category("Buy a gift") == "general"
