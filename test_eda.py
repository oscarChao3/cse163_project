"""
Oscar Chao and Leo Tran
CSE 163

Tests insurance coverage calculations, LEP ranges, and data joins
using small datasets with known results.
"""

import pandas as pd

import eda


def test_insurance_rates():
    """Checks age-specific denominators and zero-population handling."""
    data = pd.DataFrame({
        "State": ["Example", "Empty"],
        "B27010_034E": [200, 0],
        "B27010_035E": [150, 0],
        "B27010_050E": [20, 0],
        "B27010_051E": [100, 0],
        "B27010_066E": [5, 0],
    })

    result = eda.insurance_rates(data).set_index(["State", "Age"])

    assert abs(
        result.loc[("Example", "35-64"), "Coverage_Percentage"] - 90
    ) < 0.000001

    assert abs(
        result.loc[("Example", "65+"), "Coverage_Percentage"] - 95
    ) < 0.000001

    assert pd.isna(
        result.loc[("Empty", "35-64"), "Coverage_Percentage"]
    )


def test_lep_ranges():
    """Checks that ranges are calculated separately for each group."""
    data = pd.DataFrame({
        "State": ["A", "B", "A", "B"],
        "Age": ["18-64", "18-64", "65+", "65+"],
        "Language": ["Spanish"] * 4,
        "LEP_Percentage": [20, 50, 60, 70],
    })

    summary, ranges = eda.summarize_lep(data)

    assert ranges.loc[("18-64", "Spanish"), "Range"] == 30
    assert ranges.loc[("65+", "Spanish"), "Range"] == 10
    assert summary.loc[("18-64", "Spanish"), "mean"] == 35


def test_hearing_insurance_merge():
    """Checks matching by state and age despite different row orders."""
    hearing = pd.DataFrame({
        "State": ["A", "A", "B", "A"],
        "Age": ["35-64", "35-64", "35-64", "65+"],
        "Sex": ["Male", "Female", "Male", "Male"],
        "Hearing_Percentage": [4, 2, 3, 18],
    })

    insurance = pd.DataFrame({
        "State": ["A", "B", "A"],
        "Age": ["65+", "35-64", "35-64"],
        "Coverage_Percentage": [99, 80, 90],
    })

    result = eda.merge_hearing_insurance(hearing, insurance)
    result = result.set_index(["State", "Age", "Sex"])

    assert len(result) == 4
    assert result.loc[
        ("A", "35-64", "Male"), "Coverage_Percentage"
    ] == 90
    assert result.loc[
        ("A", "35-64", "Female"), "Coverage_Percentage"
    ] == 90
    assert result.loc[
        ("B", "35-64", "Male"), "Coverage_Percentage"
    ] == 80
    assert result.loc[
        ("A", "65+", "Male"), "Coverage_Percentage"
    ] == 99


def test_language_percentages():
    """Checks language composition and LEP use different denominators."""
    data = pd.DataFrame(
        0,
        index=[0],
        columns=[
            f"B16004_{number:03d}E"
            for number in range(1, 68)
        ],
    )
    data["State"] = ["Example"]

    values = {
        "B16004_024E": 100,
        "B16004_025E": 60,
        "B16004_026E": 40,
        "B16004_027E": 20,
        "B16004_028E": 10,
        "B16004_029E": 6,
        "B16004_030E": 4,
        "B16004_046E": 50,
        "B16004_047E": 30,
        "B16004_048E": 20,
        "B16004_049E": 5,
        "B16004_050E": 8,
        "B16004_051E": 4,
        "B16004_052E": 3,
    }

    for column, value in values.items():
        data[column] = value

    composition = eda.language_composition(data).set_index(
        ["Age", "Language"]
    )
    lep = eda.language_lep(data).set_index(["Age", "Language"])

    assert composition.loc[
        ("18-64", "Spanish"), "Percentage"
    ] == 40

    assert lep.loc[
        ("18-64", "Spanish"), "LEP_Percentage"
    ] == 50

    assert lep.loc[
        ("65+", "Spanish"), "LEP_Percentage"
    ] == 75

    assert pd.isna(
        lep.loc[("18-64", "Other languages"), "LEP_Percentage"]
    )


def test_actual_data():
    """Checks geography, missingness, and percentages in actual data."""
    datasets = {}

    for table in ["B16004", "B18102", "B27010", "C16001"]:
        cleaned = eda.read_data(f"data/ACSDT5Y2024.{table}.csv")
        data = eda.prepare_estimates(cleaned, table)
        datasets[table] = data

        assert len(data) == 51
        assert data["State"].is_unique
        assert "District of Columbia" in data["State"].values
        assert "Puerto Rico" not in data["State"].values
        assert "United States" not in data["State"].values
        assert not data.isna().any().any()

    states = set(datasets["B16004"]["State"])
    for data in datasets.values():
        assert set(data["State"]) == states

    composition = eda.language_composition(datasets["B16004"])
    totals = composition.groupby(
        ["State", "Age"]
    )["Percentage"].sum()

    assert len(composition) == 510
    assert composition["Percentage"].between(0, 100).all()
    assert ((totals - 100).abs() < 0.000001).all()

    lep = eda.language_lep(datasets["B16004"])
    assert len(lep) == 408
    assert not lep.isna().any().any()
    assert lep["LEP_Percentage"].between(0, 100).all()
    assert (lep["LEP_Count"] <= lep["Group_Total"]).all()


def main():
    test_insurance_rates()
    print("Insurance rates: passed")

    test_lep_ranges()
    print("LEP ranges: passed")

    test_hearing_insurance_merge()
    print("Hearing-insurance merge: passed")

    test_language_percentages()
    print("Language percentages: passed")

    test_actual_data()
    print("Actual data checks: passed")


if __name__ == "__main__":
    main()
