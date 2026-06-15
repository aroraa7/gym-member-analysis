from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
RAW_DATA_PATH = DATA_DIR / "gym_members_exercise_tracking.csv"
CLEAN_DATA_PATH = DATA_DIR / "gymmembers_exercise_New.csv"
NULL_REPORT_PATH = OUTPUT_DIR / "null_value_counts.csv"


def inches_to_feet_inches(inches):
    """Convert inches to a feet.inches float format."""
    feet = int(inches // 12)
    remaining_inches = int(inches % 12)
    return float(f"{feet}.{remaining_inches}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Read in and inspect the raw dataset.
    df = pd.read_csv(RAW_DATA_PATH)
    print(df.head(10))
    print(df.describe())
    df.info()

    dfcopy = df.copy()

    # Convert metric height and weight into imperial units.
    dfcopy["Weight_lbs"] = round(dfcopy["Weight (kg)"] * 2.20462, 2)
    dfcopy["Height_inches"] = round(dfcopy["Height (m)"] * 39.3701, 2)
    dfcopy["Height_ft"] = dfcopy["Height_inches"].apply(inches_to_feet_inches)
    dfcopy = dfcopy.drop(["Weight (kg)", "Height (m)", "Height_inches"], axis=1)

    print(dfcopy.head(10))
    dfcopy.info()

    any_null = dfcopy.isnull().values.any()
    print(f"Any null values in the data: {any_null}")

    null_counts = dfcopy.isnull().sum()
    print("\nNull value count in each column:")
    print(null_counts)

    dfcopy.to_csv(CLEAN_DATA_PATH, index=False)
    null_counts.to_csv(NULL_REPORT_PATH, header=["null_count"])
    print(f"\nCleaned data written to {CLEAN_DATA_PATH}")
    print(f"Null-value report written to {NULL_REPORT_PATH}")


if __name__ == "__main__":
    main()
