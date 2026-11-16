"""
Oscar Chao and Leo Tran
CSE 163

This program explores ACS data about language, hearing difficulty,
and health insurance coverage across states.
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


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


def language_composition(data):
    """
    Takes the prepared B16004 estimates and returns language counts
    and percentages by state and age group.
    """
    languages = [
        "English only",
        "Spanish",
        "Other Indo-European",
        "Asian and Pacific Island",
        "Other languages",
    ]

    age_groups = {
        "18-64": (24, [25, 26, 31, 36, 41]),
        "65+": (46, [47, 48, 53, 58, 63]),
    }

    results = []

    for age, (total_number, language_numbers) in age_groups.items():
        total = data[f"B16004_{total_number:03d}E"]

        for language, number in zip(languages, language_numbers):
            count = data[f"B16004_{number:03d}E"]

            group = pd.DataFrame({
                "State": data["State"],
                "Age": age,
                "Language": language,
                "Count": count,
                "Total": total,
                "Percentage": count / total.where(total > 0) * 100,
            })
            results.append(group)

    return pd.concat(results, ignore_index=True)


def summarize_composition(composition):
    """
    Takes language composition data and returns percentage totals
    by state and age, and descriptive statistics by age and language.
    """
    totals = composition.groupby(
        ["State", "Age"]
    )["Percentage"].sum()

    summary = composition.groupby(
        ["Age", "Language"]
    )["Percentage"].describe()

    return totals, summary


def plot_language_composition(composition):
    """
    Takes language composition data and saves a grouped bar chart
    comparing mean state percentages between the two age groups.
    Each state and Washington, D.C. receives equal weight.
    """
    means = composition.groupby(
        ["Language", "Age"]
    )["Percentage"].mean().unstack()

    languages = [
        "English only",
        "Spanish",
        "Other Indo-European",
        "Asian and Pacific Island",
        "Other languages",
    ]
    means = means.loc[languages, ["18-64", "65+"]]

    ax = means.plot.bar(figsize=(10, 6))
    ax.set_title("Language Composition by Age: Mean State Percentages")
    ax.set_xlabel("Language spoken at home")
    ax.set_ylabel("Mean state percentage (%)")
    ax.legend(title="Age group")

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("language_composition.png", dpi=300)
    plt.close()


def language_lep(data):
    """
    Takes prepared B16004 estimates and returns LEP counts and
    percentages by state, age group, and non-English language group.
    """
    languages = [
        "Spanish",
        "Other Indo-European",
        "Asian and Pacific Island",
        "Other languages",
    ]

    age_groups = {
        "18-64": [26, 31, 36, 41],
        "65+": [48, 53, 58, 63],
    }

    results = []

    for age, group_numbers in age_groups.items():
        for language, number in zip(languages, group_numbers):
            total = data[f"B16004_{number:03d}E"]

            # The three LEP categories follow the "very well" category.
            lep_columns = [
                f"B16004_{number + offset:03d}E"
                for offset in [2, 3, 4]
            ]
            lep_count = data[lep_columns].sum(axis=1, min_count=3)

            group = pd.DataFrame({
                "State": data["State"],
                "Age": age,
                "Language": language,
                "LEP_Count": lep_count,
                "Group_Total": total,
                "LEP_Percentage": (
                    lep_count / total.where(total > 0) * 100
                ),
            })
            results.append(group)

    return pd.concat(results, ignore_index=True)


def plot_lep_ranges(ranges):
    """
    Takes LEP ranges indexed by age and language and saves an
    interactive bar chart comparing ranges across states.
    """
    plot_data = ranges.reset_index()
    plot_data["Group"] = (
        plot_data["Age"] + " / " + plot_data["Language"]
    )

    fig = px.bar(
        plot_data,
        x="Range",
        y="Group",
        orientation="h",
        hover_data=["min", "max"],
        title="LEP Rate Ranges Across States by Age and Language",
        labels={
            "Range": "Highest minus lowest LEP rate (percentage points)",
            "Group": "Age and language group",
            "min": "Lowest LEP rate (%)",
            "max": "Highest LEP rate (%)",
        },
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=plot_data["Group"].tolist(),
        autorange="reversed",
    )
    fig.update_layout(width=1100, height=600)
    fig.write_html("language_lep_ranges.html")


def summarize_lep(lep):
    """
    Takes LEP data and returns descriptive statistics and state ranges
    for each age and language group, with ranges sorted largest first.
    """
    summary = lep.groupby(
        ["Age", "Language"]
    )["LEP_Percentage"].describe()

    ranges = summary[["min", "max"]].copy()
    ranges["Range"] = ranges["max"] - ranges["min"]
    ranges = ranges.sort_values("Range", ascending=False)

    return summary, ranges


def find_lep_extremes(lep):
    """
    Takes LEP data and returns two DataFrames containing the lowest
    and highest observed rate in each age and language group.
    If rates are tied, returns the first matching row.
    """
    grouped = lep.groupby(["Age", "Language"])["LEP_Percentage"]
    lowest = lep.loc[grouped.idxmin()].copy()
    highest = lep.loc[grouped.idxmax()].copy()

    return lowest, highest


def inspect_lep_uncertainty(data):
    """
    Takes cleaned B16004 data and returns counts and margins of error
    for the 65+ other-language group in Wyoming and Michigan.
    """
    columns = [
        "Label (Grouping)",
        "Wyoming!!Estimate",
        "Wyoming!!Margin of Error",
        "Michigan!!Estimate",
        "Michigan!!Margin of Error",
    ]

    # Group total and the three categories counted as LEP.
    result = data.iloc[[62, 64, 65, 66]][columns].copy()
    result["Label (Grouping)"] = (
        result["Label (Grouping)"].str.strip()
    )

    return result


def hearing_rates(data):
    """
    Takes prepared B18102 estimates and returns hearing difficulty
    counts and percentages by state, sex, and age group.
    """
    groups = {
        ("Male", "35-64"): ([12], [13]),
        ("Male", "65+"): ([15, 18], [16, 19]),
        ("Female", "35-64"): ([31], [32]),
        ("Female", "65+"): ([34, 37], [35, 38]),
    }

    results = []

    for (sex, age), (total_numbers, difficulty_numbers) in groups.items():
        total_columns = [
            f"B18102_{number:03d}E" for number in total_numbers
        ]
        difficulty_columns = [
            f"B18102_{number:03d}E" for number in difficulty_numbers
        ]

        total = data[total_columns].sum(
            axis=1, min_count=len(total_columns)
        )
        difficulty = data[difficulty_columns].sum(
            axis=1, min_count=len(difficulty_columns)
        )

        group = pd.DataFrame({
            "State": data["State"],
            "Sex": sex,
            "Age": age,
            "Hearing_Count": difficulty,
            "Population": total,
            "Hearing_Percentage": (
                difficulty / total.where(total > 0) * 100
            ),
        })
        results.append(group)

    return pd.concat(results, ignore_index=True)


def insurance_rates(data):
    """
    Takes prepared B27010 estimates and returns insurance coverage
    percentages by state and age group, with both sexes combined.
    """
    groups = {
        "35-64": (34, 50),
        "65+": (51, 66),
    }

    results = []

    for age, (total_number, uninsured_number) in groups.items():
        total = data[f"B27010_{total_number:03d}E"]
        uninsured = data[f"B27010_{uninsured_number:03d}E"]

        group = pd.DataFrame({
            "State": data["State"],
            "Age": age,
            "Insurance_Population": total,
            "Uninsured_Count": uninsured,
            "Coverage_Percentage": (
                (1 - uninsured / total.where(total > 0)) * 100
            ),
        })
        results.append(group)

    return pd.concat(results, ignore_index=True)


def merge_hearing_insurance(hearing, insurance):
    """
    Takes hearing rates by state, sex, and age, and insurance rates
    by state and age. Returns their inner join on state and age.
    Requires unique insurance records for each state and age group.
    """
    return hearing.merge(
        insurance,
        on=["State", "Age"],
        how="inner",
        validate="many_to_one",
    )


def summarize_hearing_insurance(data):
    """
    Takes joined hearing and insurance data and returns descriptive
    statistics for both percentages within each sex and age group.
    Insurance coverage is shared by both sexes within each state-age.
    """
    return data.groupby(["Sex", "Age"])[
        ["Hearing_Percentage", "Coverage_Percentage"]
    ].describe()


def plot_hearing_insurance(data):
    """
    Takes joined hearing and insurance data and saves one scatter
    plot per age group, with sex distinguished by color.
    Each point represents one state and sex within that age group.
    """
    for age in ["35-64", "65+"]:
        age_data = data[data["Age"] == age]
        fig, ax = plt.subplots(figsize=(9, 6))

        for sex in ["Male", "Female"]:
            group = age_data[age_data["Sex"] == sex]
            ax.scatter(
                group["Hearing_Percentage"],
                group["Coverage_Percentage"],
                label=sex,
                alpha=0.75,
            )

        ax.set_title(
            f"Hearing Difficulty and Insurance Coverage: Ages {age}"
        )
        ax.set_xlabel("Hearing difficulty within sex and age group (%)")
        ax.set_ylabel("Insurance coverage within age group, both sexes (%)")
        ax.legend(title="Sex of hearing measure")

        fig.tight_layout()
        filename = age.replace("+", "plus")
        fig.savefig(f"hearing_insurance_{filename}.png", dpi=300)
        plt.close(fig)


def main():
    tables = ["B16004", "B18102", "B27010", "C16001"]
    datasets = {}

    for table in tables:
        data = read_data(f"data/ACSDT5Y2024.{table}.csv")
        estimates = prepare_estimates(data, table)
        estimates.to_csv(f"data/{table}_estimates.csv", index=False)
        datasets[table] = estimates

    composition = language_composition(datasets["B16004"])
    totals, summary = summarize_composition(composition)

    composition.to_csv("data/language_composition.csv", index=False)
    summary.to_csv("data/language_composition_summary.csv")

    print("Language composition:", composition.shape)
    print(composition.head())
    print("\nPercentage totals for each state and age group:")
    print(totals.head(6))
    print("\nSummary of language percentages:")
    print(summary)

    plot_language_composition(composition)

    lep = language_lep(datasets["B16004"])
    lep_summary, ranges = summarize_lep(lep)
    lowest, highest = find_lep_extremes(lep)

    lep.to_csv("data/language_lep.csv", index=False)
    lep_summary.to_csv("data/language_lep_summary.csv")
    ranges.to_csv("data/language_lep_ranges.csv")
    lowest.to_csv("data/lep_lowest_states.csv", index=False)
    highest.to_csv("data/lep_highest_states.csv", index=False)

    print("\nLEP data:", lep.shape)
    print(lep.head())
    print("\nMissing values:")
    print(lep.isna().sum())
    print("\nLEP ranges across states:")
    print(ranges)
    print("\nStates with the lowest LEP percentages:")
    print(lowest.to_string(index=False))
    print("\nStates with the highest LEP percentages:")
    print(highest.to_string(index=False))

    plot_lep_ranges(ranges)

    language_data = read_data("data/ACSDT5Y2024.B16004.csv")
    uncertainty = inspect_lep_uncertainty(language_data)

    print("\nUncertainty in the 65+ other-language group:")
    print(uncertainty.to_string(index=False))
    uncertainty.to_csv("data/lep_uncertainty.csv", index=False)

    hearing = hearing_rates(datasets["B18102"])
    hearing.to_csv("data/hearing_rates.csv", index=False)

    print("\nHearing data:", hearing.shape)
    print(hearing.head())

    print("\nMissing values:")
    print(hearing.isna().sum())

    print("\nMean hearing difficulty percentages:")
    print(
        hearing.groupby(
            ["Sex", "Age"]
        )["Hearing_Percentage"].mean()
    )

    insurance = insurance_rates(datasets["B27010"])
    insurance.to_csv("data/insurance_rates.csv", index=False)

    print("\nInsurance data:", insurance.shape)
    print(insurance.head())

    print("\nMissing values:")
    print(insurance.isna().sum())

    print("\nInsurance coverage summary:")
    insurance_summary = insurance.groupby(
        "Age"
    )["Coverage_Percentage"].describe()
    print(insurance_summary)
    insurance_summary.to_csv("data/insurance_summary.csv")

    hearing_insurance = merge_hearing_insurance(hearing, insurance)
    hearing_insurance.to_csv(
        "data/hearing_insurance.csv", index=False
    )

    print("\nJoined hearing and insurance data:")
    print("Rows and columns:", hearing_insurance.shape)
    print(hearing_insurance.head().to_string(index=False))

    assert len(hearing_insurance) == len(hearing)
    assert not hearing_insurance.duplicated(
        ["State", "Sex", "Age"]
    ).any()

    print("\nMissing values:")
    print(hearing_insurance.isna().sum())

    hearing_insurance_summary = summarize_hearing_insurance(
        hearing_insurance
    )
    hearing_insurance_summary.to_csv(
        "data/hearing_insurance_summary.csv"
    )

    print("\nHearing and insurance summary:")
    print(hearing_insurance_summary.to_string())

    plot_hearing_insurance(hearing_insurance)


if __name__ == "__main__":
    main()