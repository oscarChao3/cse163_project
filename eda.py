"""
Oscar Chao and Leo Tran
CSE 163

This program explores ACS data about language, hearing difficulty,
and health insurance coverage across states.
"""

import pandas as pd


def read_data(filepath):
    """
    Reads an ACS CSV from filepath and returns a DataFrame containing
    the 50 states and Washington, D.C. Converts estimates and margins
    of error to numbers, with unrecognized values treated as missing.
    """
    data = pd.read_csv(filepath)

    excluded = [
        "United States!!Estimate",
        "United States!!Margin of Error",
        "Puerto Rico!!Estimate",
        "Puerto Rico!!Margin of Error",
    ]
    data = data.drop(columns=excluded)

    for column in data.columns[1:]:
        values = data[column].str.replace(",", "", regex=False)
        values = values.str.replace("±", "", regex=False)
        data[column] = pd.to_numeric(values, errors="coerce")

    return data


def prepare_estimates(data, table):
    """
    Takes a cleaned ACS DataFrame and its table name.
    Returns a DataFrame with one row per state and numbered estimate
    columns corresponding to the original table's row order.
    """
    estimates = data.filter(like="!!Estimate").copy()

    # Number the categories because some original labels are repeated.
    estimates.index = [
        f"{table}_{i + 1:03d}E" for i in range(len(estimates))
    ]

    estimates = estimates.T
    estimates.index = estimates.index.str.replace(
        "!!Estimate", "", regex=False
    )
    estimates.index.name = "State"

    return estimates.reset_index()


def main():
    tables = ["B16004", "B18102", "B27010", "C16001"]

    for table in tables:
        data = read_data(f"data/ACSDT5Y2024.{table}.csv")
        estimates = prepare_estimates(data, table)
        estimates.to_csv(f"data/{table}_estimates.csv", index=False)

        print(f"\n{table}")
        print("Rows and columns:", estimates.shape)
        print(estimates.iloc[:3, :5])

        if table == "B16004":
            for i, label in enumerate(data["Label (Grouping)"]):
                print(f"B16004_{i + 1:03d}E: {label.strip()}")


if __name__ == "__main__":
    main()