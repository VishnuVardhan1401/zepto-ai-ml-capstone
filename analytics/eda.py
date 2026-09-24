
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


# ============================================================
# TASK 1 - LOAD, PROFILE AND SAVE OFFLINE FALLBACK
# ============================================================

print("\n" + "=" * 70)
print("TASK 1 - LOAD AND PROFILE TITANIC DATASET")
print("=" * 70)

# Load raw Titanic dataset exactly once
df = sns.load_dataset("titanic")

# REQUIRED OFFLINE FALLBACK
# This is saved immediately after loading.
df.to_csv("titanic.csv", index=False)

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nDescriptive statistics:")
print(df.describe(include="all"))

print("\nMissing values:")
missing_percent = df.isnull().mean() * 100

missing_report = missing_percent[
    missing_percent > 0
].sort_values(ascending=False)

print(missing_report.round(2))

print("\nMissing-value report:")
for column, percentage in missing_report.items():
    print(f"{column}: {percentage:.2f}%")


# ============================================================
# TASK 2 - MISSING VALUE HANDLING
# ============================================================

print("\n" + "=" * 70)
print("TASK 2 - MISSING VALUE HANDLING")
print("=" * 70)

print("""
Missing-value strategy:

- Under 5% missing: drop affected rows.
- 5% to 30% missing: impute.
- Above 30% missing: drop the column or encode missing as a category
  when appropriate.
""")

# Store the original missing percentages before any cleaning.
original_missing = (
    df.isnull().mean() * 100
)

print("\nOriginal missing percentages:")

for column in original_missing[
    original_missing > 0
].index:

    print(
        f"{column}: "
        f"{original_missing[column]:.2f}%"
    )


# ---- AGE ----
# Age has approximately 19.87% missing values in the Titanic dataset.
# This falls between 5% and 30%, so median imputation is used.

if "age" in df.columns:

    age_missing = original_missing["age"]

    if 5 <= age_missing <= 30:

        age_median = df["age"].median()

        df["age"] = df["age"].fillna(
            age_median
        )

        print(
            f"\nAge: {age_missing:.2f}% missing "
            f"-> median imputation."
        )


# ---- EMBARKED ----
# Embarked has less than 5% missing values.
# Therefore rows with missing embarked are dropped.

if "embarked" in df.columns:

    embarked_missing = original_missing["embarked"]

    if embarked_missing < 5:

        before = len(df)

        df = df.dropna(
            subset=["embarked"]
        )

        removed = before - len(df)

        print(
            f"Embarked: {embarked_missing:.2f}% missing "
            f"-> dropped {removed} rows."
        )


# ---- DECK ----
# Deck has very high missingness (above 30%).
# Imputation would be unreliable, so the column is dropped.

if "deck" in df.columns:

    deck_missing = original_missing["deck"]

    if deck_missing > 30:

        df = df.drop(
            columns=["deck"]
        )

        print(
            f"Deck: {deck_missing:.2f}% missing "
            f"-> column dropped because missingness is too high "
            f"for reliable imputation."
        )
        
        if "embark_town" in df.columns:
            embark_town_missing = original_missing["embark_town"]

    if embark_town_missing < 5:
        df = df.dropna(subset=["embark_town"])
        print(
            f"Embark_town: {embark_town_missing:.2f}% missing "
            f"-> rows dropped."
        )


print("\nMissing values after cleaning:")

remaining_missing = df.isnull().sum()

print(
    remaining_missing[
        remaining_missing > 0
    ]
)

print("\nCleaned dataset shape:")
print(df.shape)


# ============================================================
# TASK 3 - UNIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("TASK 3 - UNIVARIATE ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# AGE HISTOGRAM
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["age"],
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# AGE BOX PLOT
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["age"]
)

plt.title("Age Box Plot")
plt.xlabel("Age")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# FARE HISTOGRAM
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["fare"],
    kde=True
)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# FARE BOX PLOT
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    x=df["fare"]
)

plt.title("Fare Box Plot")
plt.xlabel("Fare")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# IQR OUTLIER FUNCTION
# ------------------------------------------------------------

def calculate_iqr_outliers(data, column):

    q1 = data[column].quantile(0.25)

    q3 = data[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr

    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (data[column] < lower_bound)
        |
        (data[column] > upper_bound)
    )

    outlier_count = outlier_mask.sum()

    return (
        q1,
        q3,
        iqr,
        lower_bound,
        upper_bound,
        outlier_count
    )


# ------------------------------------------------------------
# AGE OUTLIERS
# ------------------------------------------------------------

age_q1, age_q3, age_iqr, age_lower, age_upper, age_outliers = (
    calculate_iqr_outliers(
        df,
        "age"
    )
)

print("\nAGE IQR RESULTS")

print(f"Q1: {age_q1:.4f}")
print(f"Q3: {age_q3:.4f}")
print(f"IQR: {age_iqr:.4f}")
print(f"Lower bound: {age_lower:.4f}")
print(f"Upper bound: {age_upper:.4f}")
print(f"Number of outliers: {age_outliers}")


# ------------------------------------------------------------
# FARE OUTLIERS
# ------------------------------------------------------------

fare_q1, fare_q3, fare_iqr, fare_lower, fare_upper, fare_outliers = (
    calculate_iqr_outliers(
        df,
        "fare"
    )
)

print("\nFARE IQR RESULTS")

print(f"Q1: {fare_q1:.4f}")
print(f"Q3: {fare_q3:.4f}")
print(f"IQR: {fare_iqr:.4f}")
print(f"Lower bound: {fare_lower:.4f}")
print(f"Upper bound: {fare_upper:.4f}")
print(f"Number of outliers: {fare_outliers}")


# ------------------------------------------------------------
# FARE MEAN, MEDIAN AND MODE
# ------------------------------------------------------------

fare_mean = df["fare"].mean()

fare_median = df["fare"].median()

fare_mode = df["fare"].mode().iloc[0]

fare_skew = df["fare"].skew()

print("\nFARE STATISTICS")

print(f"Mean: {fare_mean:.4f}")

print(f"Median: {fare_median:.4f}")

print(f"Mode: {fare_mode:.4f}")

print(f"Skewness: {fare_skew:.4f}")


if fare_mean > fare_median > fare_mode:

    fare_distribution = "right-skewed"

elif fare_mean < fare_median < fare_mode:

    fare_distribution = "left-skewed"

else:

    fare_distribution = "not clearly determined by mean/median/mode ordering"


print(
    f"Fare distribution: {fare_distribution}"
)

print("""
INTERPRETATION:
The ordering of mean, median and mode is used as the requested
indicator of skewness. The positive skewness value provides an
additional numerical check of the distribution shape.
""")


# ============================================================
# TASK 4 - BIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("TASK 4 - BIVARIATE ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# SURVIVAL RATE BY SEX
# Using boolean masking
# ------------------------------------------------------------

print("\nSURVIVAL RATE BY SEX")

male_mask = df["sex"] == "male"

female_mask = df["sex"] == "female"

male_survival = df.loc[
    male_mask,
    "survived"
].mean()

female_survival = df.loc[
    female_mask,
    "survived"
].mean()

print(
    f"Male survival rate: "
    f"{male_survival:.4f}"
)

print(
    f"Female survival rate: "
    f"{female_survival:.4f}"
)


# ------------------------------------------------------------
# SURVIVAL RATE BY PCLASS
# ------------------------------------------------------------

print("\nSURVIVAL RATE BY PCLASS")

for pclass in sorted(
    df["pclass"].dropna().unique()
):

    pclass_mask = (
        df["pclass"] == pclass
    )

    survival_rate = df.loc[
        pclass_mask,
        "survived"
    ].mean()

    print(
        f"Class {pclass}: "
        f"{survival_rate:.4f}"
    )


# ------------------------------------------------------------
# SURVIVAL RATE BY SEX AND PCLASS
# Boolean masking with &
# ------------------------------------------------------------

print("\nSURVIVAL RATE BY SEX AND PCLASS")

for sex in sorted(
    df["sex"].dropna().unique()
):

    for pclass in sorted(
        df["pclass"].dropna().unique()
    ):

        mask = (
            (df["sex"] == sex)
            &
            (df["pclass"] == pclass)
        )

        group = df.loc[
            mask,
            "survived"
        ]

        if len(group) > 0:

            print(
                f"{sex}, Class {pclass}: "
                f"{group.mean():.4f}"
            )


# ------------------------------------------------------------
# EXACT SIX-COLUMN CORRELATION MATRIX
# ------------------------------------------------------------

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation_matrix = df[
    correlation_columns
].corr()

print("\nEXACT 6 x 6 CORRELATION MATRIX")

print(
    correlation_matrix.round(4)
)


# ------------------------------------------------------------
# CORRELATION HEATMAP
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    square=True
)

plt.title(
    "Titanic Correlation Matrix"
)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# FIND TOP TWO ABSOLUTE OFF-DIAGONAL CORRELATIONS
# ------------------------------------------------------------

correlation_pairs = []

for i in range(
    len(correlation_columns)
):

    for j in range(
        i + 1,
        len(correlation_columns)
    ):

        column_1 = correlation_columns[i]

        column_2 = correlation_columns[j]

        value = correlation_matrix.loc[
            column_1,
            column_2
        ]

        correlation_pairs.append(
            (
                column_1,
                column_2,
                value,
                abs(value)
            )
        )


correlation_pairs.sort(
    key=lambda x: x[3],
    reverse=True
)


print("\nTWO STRONGEST CORRELATIONS")

for pair in correlation_pairs[:2]:

    column_1, column_2, value, absolute_value = pair

    print(
        f"{column_1} vs {column_2}: "
        f"correlation = {value:.4f}, "
        f"|correlation| = {absolute_value:.4f}"
    )

print("""
INTERPRETATION:
The two pairs above are selected strictly by ranking the absolute
off-diagonal correlation coefficients. A positive coefficient means
the two variables tend to increase together, while a negative
coefficient indicates an inverse relationship.
""")


# ============================================================
# TASK 5 - MULTIVARIATE DATA STORY
# ============================================================

print("\n" + "=" * 70)
print("TASK 5 - MULTIVARIATE DATA STORY")
print("=" * 70)


# ------------------------------------------------------------
# CHART 1
# Survival by Sex and Passenger Class
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title(
    "Survival Rate by Sex and Passenger Class"
)

plt.xlabel("Passenger Class")

plt.ylabel("Survival Rate")

plt.tight_layout()

plt.show()

print("""
CHART 1 INTERPRETATION:
Female passengers generally had higher survival rates than male
passengers across passenger classes. Higher passenger classes also
generally had better survival outcomes, showing the combined effect
of sex and socioeconomic position.
""")


# ------------------------------------------------------------
# CHART 2
# Fare and Survival
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="survived",
    y="fare"
)

plt.title(
    "Fare Distribution by Survival"
)

plt.xlabel(
    "Survived (0 = No, 1 = Yes)"
)

plt.ylabel("Fare")

plt.tight_layout()

plt.show()

print("""
CHART 2 INTERPRETATION:
Survivors generally have higher fare values than non-survivors.
This supports the idea that passenger class and socioeconomic
position were associated with survival probability.
""")


# ------------------------------------------------------------
# CHART 3
# Age and Fare by Survival
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived"
)

plt.title(
    "Age vs Fare by Survival"
)

plt.xlabel("Age")

plt.ylabel("Fare")

plt.tight_layout()

plt.show()

print("""
CHART 3 INTERPRETATION:
Age and fare overlap between survivors and non-survivors, so neither
variable completely separates the two outcomes by itself. The chart
shows why multiple passenger characteristics are useful for explaining
survival rather than relying on a single feature.
""")


# ------------------------------------------------------------
# CHART 4
# Age, Sex and Survival
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="sex",
    y="age",
    hue="survived"
)

plt.title(
    "Age Distribution by Sex and Survival"
)

plt.xlabel("Sex")

plt.ylabel("Age")

plt.tight_layout()

plt.show()

print("""
CHART 4 INTERPRETATION:
The age distributions differ between survival groups within the sex
categories. Together with the earlier charts, this suggests that
survival was related to several interacting passenger characteristics.
""")


# ============================================================
# TASK 6 - Z-SCORE STANDARDIZATION SANITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("TASK 6 - Z-SCORE STANDARDIZATION")
print("=" * 70)


# This is EDA-only standardization.
# It is NOT used by modeling.py.

eda_scaler = StandardScaler()

standardized_values = eda_scaler.fit_transform(
    df[
        ["age", "fare"]
    ]
)

standardized_df = pd.DataFrame(
    standardized_values,
    columns=[
        "age_z",
        "fare_z"
    ]
)


print("\nBEFORE STANDARDIZATION")

print(
    df[
        ["age", "fare"]
    ].agg(
        ["mean", "std"]
    )
)


print("\nAFTER STANDARDIZATION")

print(
    standardized_df.agg(
        ["mean", "std"]
    )
)


print("\nPopulation standard deviations after scaling:")

print(
    standardized_df.std(
        ddof=0
    ).round(6)
)


print("""
INTERPRETATION:
The standardized age and fare variables have approximately mean 0
and population standard deviation 1. This confirms that the z-score
transformation was applied correctly. This EDA transformation is only
a sanity check and does not feed into the predictive modeling pipeline.
""")