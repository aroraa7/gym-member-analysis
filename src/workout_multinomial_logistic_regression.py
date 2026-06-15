from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "gymmembers_exercise_New.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
SUMMARY_PATH = OUTPUT_DIR / "multinomial_logistic_regression_summary.txt"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    caloriesdf = df.copy()
    caloriesdf["Calories_Burned_Per_Hour"] = (
        caloriesdf["Calories_Burned"] / caloriesdf["Session_Duration (hours)"]
    )

    print("New Dataframe with Response Variable; Calories_Burned_Per_Hour:\n")
    print(caloriesdf)

    workoutdf = caloriesdf.copy()

    workout_encoder = LabelEncoder()
    workoutdf["Workout_Type_Encoded"] = workout_encoder.fit_transform(
        workoutdf["Workout_Type"]
    )

    label_mapping = dict(
        zip(workout_encoder.classes_, workout_encoder.transform(workout_encoder.classes_))
    )
    print("Workout type encoding:")
    for label, encoded_value in label_mapping.items():
        print(f"{label} = {encoded_value}")
    print()

    X = workoutdf[["Calories_Burned_Per_Hour"]]
    y = workoutdf["Workout_Type_Encoded"]
    X = sm.add_constant(X)

    mnlogit_model = sm.MNLogit(y, X)
    result = mnlogit_model.fit()

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
