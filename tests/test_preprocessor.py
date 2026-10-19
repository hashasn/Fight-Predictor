import numpy as np
import pandas as pd
import pytest
from preprocessor import mmss_to_seconds, height_to_inches, reach_to_inches, parse_of, get_result, method_group

# mmss_to_seconds tests

def test_mmss_to_seconds():
    #convert a 'm:ss' string into total seconds
    # invalid or missing values should return NaN

    #valid times
    assert mmss_to_seconds("4:32") == 272
    assert mmss_to_seconds("0:00") == 0
    assert mmss_to_seconds("5:00") == 300
    assert mmss_to_seconds(" 1:05 ") == 65

    #invalid times
    assert np.isnan(mmss_to_seconds("abc"))
    assert np.isnan(mmss_to_seconds("4:5"))    # seconds must be 2 digits
    assert np.isnan(mmss_to_seconds(""))
    assert np.isnan(mmss_to_seconds(None))
    assert np.isnan(mmss_to_seconds(np.nan))

# height_to_inches

def test_height_to_inches():
    """
    convert a height string like 5' 11" into inches.
    invalid or missing values should return nan
    """
    # valid heights
    assert height_to_inches("5' 11\"") == 71
    assert height_to_inches("6' 0\"") == 72
    assert height_to_inches("5'9\"") == 69      # no space after the '

    # invalid heights
    assert np.isnan(height_to_inches("--"))
    assert np.isnan(height_to_inches("6'"))     # missing inches
    assert np.isnan(height_to_inches(None))
    assert np.isnan(height_to_inches(np.nan))


# reach_to_inches

def test_reach_to_inches():
    """
    convert a reach string like 72" into numeric inches.
    invalid or missing values should return nan
    """
    # valid reaches
    assert reach_to_inches('72"') == 72.0
    assert reach_to_inches('74.5"') == 74.5
    assert reach_to_inches("70") == 70.0        # no " symbol

    # invalid reaches
    assert np.isnan(reach_to_inches("--"))
    assert np.isnan(reach_to_inches(""))
    assert np.isnan(reach_to_inches(None))


# parse_of

def test_parse_of():
    """
    convert a stat string like '45 of 102' into (landed, attempted, rate).
    invalid or missing values should return (nan, nan, nan)
    """
    # valid stats
    landed, attempted, rate = parse_of("45 of 102")
    assert landed == 45
    assert attempted == 102
    assert rate == pytest.approx(45 / 102)

    landed, attempted, rate = parse_of("10 of 10")
    assert rate == 1.0

    # 0 attempts should give a rate of 0, not a divide by zero error
    landed, attempted, rate = parse_of("0 of 0")
    assert landed == 0
    assert attempted == 0
    assert rate == 0.0

    # invalid stats
    for value in ["--", "abc", "x of y", "", None, np.nan]:
        landed, attempted, rate = parse_of(value)
        assert np.isnan(landed)
        assert np.isnan(attempted)
        assert np.isnan(rate)


# get_result

def test_get_result():
    """
    result from one fighter's perspective.
    1 -> won, 0 -> lost, nan -> draw / no contest / unknown
    """
    # win and loss
    assert get_result("Jean Silva", "Jean Silva") == 1
    assert get_result("Jean Silva", "Jose Delgado") == 0
    assert get_result(" Jean Silva ", "Jean Silva") == 1     # whitespace is stripped

    # draw, no contest or missing winner
    for winner in ["DRAW", "draw", "NO CONTEST", "NC", "", None, np.nan]:
        assert np.isnan(get_result(winner, "Jean Silva"))


# method_group

def test_method_group():
    """
    collapse detailed method labels into ko / sub / dec / other
    """
    assert method_group("KO/TKO") == "ko"
    assert method_group("Submission") == "sub"
    assert method_group("Decision - Unanimous") == "dec"
    assert method_group("Decision - Split") == "dec"
    assert method_group("Decision - Majority") == "dec"

    # anything else falls into other
    assert method_group("DQ") == "other"
    assert method_group("Overturned") == "other"
    assert method_group("") == "other"
    assert method_group(None) == "other"
    assert method_group("TKO - Doctor's Stoppage") == "other"