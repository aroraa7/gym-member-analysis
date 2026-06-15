from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "gymmembers_exercise_New.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
SUMMARY_PATH = OUTPUT_DIR / "gender_logistic_regression_summary.txt"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    caloriesdf = df.copy()
    caloriesdf["Calories_Burned_Per_Hour"] = (
        caloriesdf["Calories_Burned"] / caloriesdf["Session_Duration (hours)"]
    )

    print("New Dataframe with Response Variable; Calories_Burned_Per_Hour:\n")
    print(caloriesdf)

    genderdf = caloriesdf.copy()
    gender_mapping = {"Male": 1, "Female": 0}
    genderdf["Gender_Con"] = genderdf["Gender"].map(gender_mapping)

    X = genderdf[["Calories_Burned_Per_Hour"]]
    y = genderdf["Gender_Con"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train_sm = sm.add_constant(X_train)

    logit_model = sm.Logit(y_train, X_train_sm)
    result = logit_model.fit()

    summary = str(result.summary())
    coefficients = result.params
    standard_errors = result.bse
    p_values = result.pvalues

    details = (
        "\n\nCoefficients:\n"
        f"{coefficients.to_string()}\n\n"
        "Standard Errors:\n"
        f"{standard_errors.to_string()}\n\n"
        "P-values:\n"
        f"{p_values.to_string()}\n"
    )

    print(summary)
    print(details)
    SUMMARY_PATH.write_text(summary + details)
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
