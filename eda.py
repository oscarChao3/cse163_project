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
    Returns a DataFrame with one row per state and numbered
    estimate columns.
    """
    estimates = data.filter(like="!!Estimate")

    columns = []
    for i in range(len(estimates)):
        columns.append(f"{table}_{i + 1:03d}E")

    estimates = estimates.T
    estimates.columns = columns
    estimates.index = estimates.index.str.replace("!!Estimate", "")
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
    Takes language composition data and returns descriptive statistics
    for percentages by age and language.
    """
    return composition.groupby(
        ["Age", "Language"]
    )["Percentage"].describe()


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
            lep_count = data[lep_columns].sum(axis=1)

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


def overall_lep_rates(data):
    """
    Takes prepared C16001 estimates and returns overall LEP counts
    and percentages for each state among people ages 5 and older.
    """
    lep_columns = [
        f"C16001_{number:03d}E"
        for number in range(5, 39, 3)
    ]

    lep_count = data[lep_columns].sum(axis=1)
    total = data["C16001_001E"]

    return pd.DataFrame({
        "State": data["State"],
        "LEP_Count": lep_count,
        "Language_Population": total,
        "LEP_Percentage": (
            lep_count / total.where(total > 0) * 100
        ),
    })


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

        total = data[total_columns].sum(axis=1)
        difficulty = data[difficulty_columns].sum(axis=1)

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


def overall_insurance_rates(data):
    """
    Takes prepared B27010 estimates and returns insurance coverage
    percentages for each state across all ages.
    """
    uninsured_columns = [
        "B27010_017E",
        "B27010_033E",
        "B27010_050E",
        "B27010_066E",
    ]

    uninsured = data[uninsured_columns].sum(axis=1)
    total = data["B27010_001E"]

    return pd.DataFrame({
        "State": data["State"],
        "Insurance_Population": total,
        "Uninsured_Count": uninsured,
        "Coverage_Percentage": (
            (1 - uninsured / total.where(total > 0)) * 100
        ),
    })


def merge_hearing_insurance(hearing, insurance):
    """
    Takes hearing rates by state, sex, and age, and insurance rates
    by state and age. Returns their inner join on state and age.
    """
    return hearing.merge(
        insurance,
        on=["State", "Age"],
        how="inner",
    )


def merge_language_insurance(lep, insurance):
    """
    Takes overall LEP and insurance data and returns their inner join
    on state.
    """
    return lep.merge(
        insurance,
        on="State",
        how="inner",
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


def plot_language_insurance(data):
    """
    Takes joined language and insurance data and saves a scatter
    plot of state LEP percentages against insurance coverage.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        data["LEP_Percentage"],
        data["Coverage_Percentage"],
        alpha=0.75,
    )

    ax.set_title("Language Barriers and Insurance Coverage Across States")
    ax.set_xlabel("Limited English proficiency, ages 5+ (%)")
    ax.set_ylabel("Insurance coverage, all ages (%)")

    fig.tight_layout()
    fig.savefig("language_insurance_scatter.png", dpi=300)
    plt.close(fig)


def plot_language_insurance_distributions(data):
    """
    Takes joined language and insurance data and saves histograms
    showing the distribution of each percentage across states.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].hist(
        data["LEP_Percentage"], bins=10, edgecolor="white"
    )
    axes[0].set_title("LEP Rates, Ages 5+")
    axes[0].set_xlabel("Limited English proficiency (%)")
    axes[0].set_ylabel("Number of states and D.C.")

    axes[1].hist(
        data["Coverage_Percentage"],
        bins=10,
        edgecolor="white",
        color="orange",
    )
    axes[1].set_title("Insurance Coverage, All Ages")
    axes[1].set_xlabel("Insurance coverage (%)")
    axes[1].set_ylabel("Number of states and D.C.")

    fig.suptitle("Distribution of State LEP and Insurance Rates")
    fig.tight_layout()
    fig.savefig("language_insurance_distributions.png", dpi=300)
    plt.close(fig)


def categorical_counts(data, columns):
    """
    Takes a DataFrame and a list of categorical column names.
    Returns each column's unique values and their counts,
    including missing values if present.
    """
    results = []

    for column in columns:
        counts = data[column].value_counts(dropna=False)
        result = pd.DataFrame({
            "Variable": column,
            "Value": counts.index,
            "Count": counts,
        })
        results.append(result)

    return pd.concat(results, ignore_index=True)


def main():
    tables = ["B16004", "B18102", "B27010", "C16001"]
    datasets = {}

    for table in tables:
        data = read_data(f"data/ACSDT5Y2024.{table}.csv")
        estimates = prepare_estimates(data, table)
        datasets[table] = estimates

        print(f"\n{table}: cleaned rows and columns", data.shape)
        print("Missing numeric values:", data.iloc[:, 1:].isna().sum().sum())
        print("Rows and columns after reshaping:", estimates.shape)

    composition = language_composition(datasets["B16004"])
    composition_summary = summarize_composition(composition)

    print("\nLanguage composition:", composition.shape)
    print("Missing values:", composition.isna().sum().sum())
    print(composition_summary)
    composition_summary.to_csv("data/language_composition_summary.csv")

    composition_counts = categorical_counts(
        composition, ["State", "Age", "Language"]
    )
    composition_counts.to_csv(
        "data/composition_category_counts.csv", index=False
    )
    plot_language_composition(composition)

    lep = language_lep(datasets["B16004"])
    lep_summary, ranges = summarize_lep(lep)

    print("\nLEP data:", lep.shape)
    print("Missing values:", lep.isna().sum().sum())
    print(lep_summary)
    print("\nLEP ranges across states:")
    print(ranges)
    lep_summary.to_csv("data/language_lep_summary.csv")
    ranges.to_csv("data/language_lep_ranges.csv")

    lep_counts = categorical_counts(lep, ["State", "Age", "Language"])
    lep_counts.to_csv("data/lep_category_counts.csv", index=False)
    plot_lep_ranges(ranges)

    hearing = hearing_rates(datasets["B18102"])
    insurance = insurance_rates(datasets["B27010"])
    hearing_insurance = merge_hearing_insurance(hearing, insurance)

    hearing_summary = summarize_hearing_insurance(hearing_insurance)
    print("\nJoined hearing and insurance data:", hearing_insurance.shape)
    print("Missing values:", hearing_insurance.isna().sum().sum())
    print(hearing_summary)
    hearing_summary.to_csv("data/hearing_insurance_summary.csv")

    hearing_counts = categorical_counts(
        hearing_insurance, ["State", "Sex", "Age"]
    )
    hearing_counts.to_csv(
        "data/hearing_insurance_category_counts.csv", index=False
    )
    plot_hearing_insurance(hearing_insurance)

    overall_lep = overall_lep_rates(datasets["C16001"])
    overall_insurance = overall_insurance_rates(datasets["B27010"])
    language_insurance = merge_language_insurance(
        overall_lep, overall_insurance
    )

    percentages = language_insurance[
        ["LEP_Percentage", "Coverage_Percentage"]
    ]
    language_summary = percentages.describe()
    print("\nJoined language and insurance data:", language_insurance.shape)
    print("Missing values:", language_insurance.isna().sum().sum())
    print(language_summary)
    language_summary.to_csv("data/language_insurance_summary.csv")

    language_counts = categorical_counts(language_insurance, ["State"])
    language_counts.to_csv(
        "data/language_insurance_category_counts.csv", index=False
    )
    plot_language_insurance(language_insurance)
    plot_language_insurance_distributions(language_insurance)


if __name__ == "__main__":
    main()
