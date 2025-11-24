# titanic_hypothesis_project.py

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------
import pandas as pd
import seaborn as sns

from scipy.stats import chi2_contingency, ttest_ind, f_oneway, pearsonr

ALPHA = 0.05  # significance level for all tests


# -----------------------------
# 2. LOAD DATA
# -----------------------------
# Seaborn has a built-in Titanic dataset.
# It will download the data automatically.
df = sns.load_dataset("titanic")

print("First 5 rows of the raw dataset:")
print(df.head())
print("\nShape (rows, columns):", df.shape)
print("\nColumns:", df.columns.tolist())

# -----------------------------
# 3. SELECT IMPORTANT COLUMNS
# -----------------------------
# We keep only the columns we need for our analysis.
df = df[["survived", "sex", "pclass", "age", "fare", "embarked", "sibsp", "parch"]]

print("\nMissing values BEFORE cleaning:")
print(df.isnull().sum())

# -----------------------------
# 4. BASIC CLEANING
# -----------------------------
# Fill missing age with median age
df["age"] = df["age"].fillna(df["age"].median())

# Fill missing embarked with most common value (mode)
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Fill missing fare with median fare
df["fare"] = df["fare"].fillna(df["fare"].median())

# Drop any row where survived, sex, or pclass is missing (very important for tests)
df = df.dropna(subset=["survived", "sex", "pclass"])

print("\nMissing values AFTER cleaning:")
print(df.isnull().sum())

# -----------------------------
# 5. FEATURE ENGINEERING
# -----------------------------
# Family size = siblings/spouse + parents/children + 1 (the person)
df["family_size"] = df["sibsp"] + df["parch"] + 1

# Is the passenger alone?
df["is_alone"] = df["family_size"].apply(lambda x: 1 if x == 1 else 0)

# Create age groups
# child: 0–12, teen: 13–19, adult: 20–59, senior: 60+
df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 12, 19, 59, 100],
    labels=["child", "teen", "adult", "senior"],
    right=True,
    include_lowest=True
)

print("\nSample of engineered columns:")
print(df[["age", "family_size", "is_alone", "age_group"]].head())

# -----------------------------
# HELPER: PRINT TEST RESULT
# -----------------------------
def print_significance(p_value: float, alpha: float = ALPHA):
    """Helper function to print if result is significant or not."""
    if p_value < alpha:
        print(f"👉 p-value = {p_value:.6f} < {alpha} → statistically SIGNIFICANT")
    else:
        print(f"👉 p-value = {p_value:.6f} ≥ {alpha} → NOT statistically significant")


print("\n" + "=" * 60)
print("HYPOTHESIS TESTS ON TITANIC DATA")
print("=" * 60)

# ---------------------------------------------------
# TEST 1: Gender vs Survival  (Chi-square test)
# ---------------------------------------------------
print("\nTEST 1: Gender vs Survival (Chi-square)")
print("H0: Gender and survival are independent (no effect).")
print("H1: Gender and survival are related (gender affects survival).")

gender_table = pd.crosstab(df["sex"], df["survived"])
print("\nGender vs Survival table:")
print(gender_table)

chi2, p, dof, expected = chi2_contingency(gender_table)

print("\nChi-square:", chi2)
print_significance(p)

# ---------------------------------------------------
# TEST 2: Passenger Class vs Survival (Chi-square)
# ---------------------------------------------------
print("\nTEST 2: Passenger Class vs Survival (Chi-square)")
print("H0: Passenger class does NOT affect survival.")
print("H1: Passenger class DOES affect survival.")

pclass_table = pd.crosstab(df["pclass"], df["survived"])
print("\nPclass vs Survival table:")
print(pclass_table)

chi2, p, dof, expected = chi2_contingency(pclass_table)

print("\nChi-square:", chi2)
print_significance(p)

# ---------------------------------------------------
# TEST 3: Embarked (Port) vs Survival (Chi-square)
# ---------------------------------------------------
print("\nTEST 3: Embarked (Port) vs Survival (Chi-square)")
print("H0: Port of embarkation does NOT affect survival.")
print("H1: Port of embarkation DOES affect survival.")

embarked_table = pd.crosstab(df["embarked"], df["survived"])
print("\nEmbarked vs Survival table:")
print(embarked_table)

chi2, p, dof, expected = chi2_contingency(embarked_table)

print("\nChi-square:", chi2)
print_significance(p)

# ---------------------------------------------------
# TEST 4: Age (Survived vs Not Survived) - t-test
# ---------------------------------------------------
print("\nTEST 4: Age difference (t-test)")
print("H0: Average age of survived = average age of not survived.")
print("H1: Average ages are different.")

survived_age = df[df["survived"] == 1]["age"]
not_survived_age = df[df["survived"] == 0]["age"]

print("\nAverage age (survived):", survived_age.mean())
print("Average age (not survived):", not_survived_age.mean())

t_stat, p = ttest_ind(survived_age, not_survived_age, equal_var=False)

print("\nT-statistic:", t_stat)
print_significance(p)

# ---------------------------------------------------
# TEST 5: Fare (Survived vs Not Survived) - t-test
# ---------------------------------------------------
print("\nTEST 5: Fare difference (t-test)")
print("H0: Average fare of survived = average fare of not survived.")
print("H1: Average fares are different.")

survived_fare = df[df["survived"] == 1]["fare"]
not_survived_fare = df[df["survived"] == 0]["fare"]

print("\nAverage fare (survived):", survived_fare.mean())
print("Average fare (not survived):", not_survived_fare.mean())

t_stat, p = ttest_ind(survived_fare, not_survived_fare, equal_var=False)

print("\nT-statistic:", t_stat)
print_significance(p)

# ---------------------------------------------------
# TEST 6: Family Size vs Survival (t-test)
# ---------------------------------------------------
print("\nTEST 6: Family Size (Survived vs Not Survived) - t-test")
print("H0: Average family size is the same for survived and not survived.")
print("H1: Average family size is different.")

survived_family = df[df["survived"] == 1]["family_size"]
not_survived_family = df[df["survived"] == 0]["family_size"]

print("\nAverage family size (survived):", survived_family.mean())
print("Average family size (not survived):", not_survived_family.mean())

t_stat, p = ttest_ind(survived_family, not_survived_family, equal_var=False)

print("\nT-statistic:", t_stat)
print_significance(p)

# ---------------------------------------------------
# TEST 7: Is Alone vs Survival (Chi-square)
# ---------------------------------------------------
print("\nTEST 7: Is Alone vs Survival (Chi-square)")
print("H0: Being alone does NOT affect survival.")
print("H1: Being alone DOES affect survival.")

alone_table = pd.crosstab(df["is_alone"], df["survived"])
print("\nIs alone vs Survival table:")
print(alone_table)

chi2, p, dof, expected = chi2_contingency(alone_table)

print("\nChi-square:", chi2)
print_significance(p)

# ---------------------------------------------------
# TEST 8: Ticket Class vs Fare (ANOVA)
# ---------------------------------------------------
print("\nTEST 8: Ticket Class vs Fare (ANOVA)")
print("H0: Average fare is the same for all classes.")
print("H1: At least one class has a different average fare.")

fare_class_1 = df[df["pclass"] == 1]["fare"]
fare_class_2 = df[df["pclass"] == 2]["fare"]
fare_class_3 = df[df["pclass"] == 3]["fare"]

f_stat, p = f_oneway(fare_class_1, fare_class_2, fare_class_3)

print("\nF-statistic:", f_stat)
print_significance(p)

# ---------------------------------------------------
# TEST 9: Age Group vs Survival (Chi-square)
# ---------------------------------------------------
print("\nTEST 9: Age Group vs Survival (Chi-square)")
print("H0: Age group does NOT affect survival.")
print("H1: Age group DOES affect survival.")

age_group_table = pd.crosstab(df["age_group"], df["survived"])
print("\nAge group vs Survival table:")
print(age_group_table)

chi2, p, dof, expected = chi2_contingency(age_group_table)

print("\nChi-square:", chi2)
print_significance(p)

# ---------------------------------------------------
# TEST 10: Correlation between Age and Fare (Pearson)
# ---------------------------------------------------
print("\nTEST 10: Correlation between Age and Fare (Pearson correlation)")
print("H0: There is NO linear relationship between age and fare.")
print("H1: There IS a linear relationship between age and fare.")

corr, p = pearsonr(df["age"], df["fare"])

print("\nCorrelation coefficient (r):", corr)
print_significance(p)

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)
