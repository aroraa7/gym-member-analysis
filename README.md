# Fitness Data Analysis: Logistic Regression

## Overview
This project analyzes gym member demographic, fitness, and health data to explore whether calorie burn is related to gender and workout type.

The analysis uses:
- Data cleaning to convert height and weight fields into imperial units.
- Binary logistic regression to model gender from calories burned per hour.
- Multinomial logistic regression to model workout type from calories burned per hour.

## Project Structure
```text
.
|-- data/
|   |-- gym_members_exercise_tracking.csv
|   `-- gymmembers_exercise_New.csv
|-- output/
|   `-- .gitkeep
|-- src/
|   |-- clean_data.py
|   |-- gender_logistic_regression.py
|   `-- workout_multinomial_logistic_regression.py
|-- requirements.txt
`-- README.md
```

## Setup
Create and activate a virtual environment, then install the project dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Analysis
Run the scripts from the project root.

```bash
python3 src/clean_data.py
python3 src/gender_logistic_regression.py
python3 src/workout_multinomial_logistic_regression.py
```

Generated reports are written to `output/`.

## Dataset
The dataset was retrieved from Kaggle: [Gym Members Exercise Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset).

It includes:
- Demographics: age, gender, BMI
- Fitness metrics: calories burned, session duration, average/max/resting heart rate
- Health metrics: fat percentage, water intake
- Workout patterns: frequency, workout type, experience level

## Analysis Summary
### Gender Prediction
Goal: test whether `Calories_Burned_Per_Hour` predicts gender, encoded as Male = 1 and Female = 0.

The model found a statistically significant relationship, but the effect size was small. Calories burned per hour alone does not explain much variation in gender.

### Workout Type Prediction
Goal: test whether `Calories_Burned_Per_Hour` predicts workout type across Cardio, HIIT, Strength, and Yoga.

The model did not find a significant relationship between calories burned per hour and workout type. Other variables, such as experience level, session duration, workout frequency, or personal goals, may be more useful predictors.

## Next Steps
- Add predictors such as experience level, workout frequency, and heart-rate metrics.
- Explore interaction effects between session duration and calorie burn.
- Compare logistic regression results with decision trees or random forests.
- Add notebooks or visualizations for exploratory analysis.
