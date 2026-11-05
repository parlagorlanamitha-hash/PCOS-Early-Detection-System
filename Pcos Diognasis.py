
"""
PCOS EARLY DETECTION SYSTEM
Random Forest + Decision Tree Classifier
Evaluation: Accuracy, Precision, Recall and F1-Score
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ---------- 1. Load Dataset ----------

FILE_PATH = r"C:\Users\P.Namitha\Downloads\pcos_dataset.csv"

df = pd.read_csv(FILE_PATH)
df.columns = df.columns.str.strip()

target_col = "PCOS_Status"

df = df.dropna(subset=[target_col]).copy()
df[target_col] = df[target_col].astype(str).str.strip().str.title()
df = df[df[target_col].isin(["Yes", "No"])].copy()

X = df.drop(columns=[target_col])
y = df[target_col]

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

print("=" * 60)
print("PCOS EARLY DETECTION SYSTEM")
print("=" * 60)
print("Dataset loaded successfully!")
print("Total records:", len(df))
print("Total features:", len(X.columns))
print("Target column:", target_col)
print("Algorithms: Random Forest and Decision Tree")

# ---------- 2. Preprocessing ----------

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_cols),
    ("categorical", categorical_transformer, categorical_cols)
])

# ---------- 3. Split Dataset ----------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------- 4. Create Models ----------

models = {
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Decision Tree": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ))
    ])
}

# ---------- 5. Train and Compare Models ----------

trained_models = {}
results = []

print("\nComparing Machine Learning Models...")
print("-" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    trained_models[name] = model

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions,
        average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions,
        average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions,
        average="macro", zero_division=0
    )

    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    print(f"\n{name}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

comparison = pd.DataFrame(results)

# ---------- 6. Best Model ----------

best = comparison.sort_values(
    by=["F1-Score", "Accuracy"],
    ascending=False
).iloc[0]

print("\n" + "=" * 60)
print("MODEL COMPARISON RESULT")
print("=" * 60)

print(comparison.to_string(
    index=False,
    formatters={
        "Accuracy": "{:.4f}".format,
        "Precision": "{:.4f}".format,
        "Recall": "{:.4f}".format,
        "F1-Score": "{:.4f}".format
    }
))

print("\nBest Model:", best["Algorithm"])
print(f"Best F1-Score: {best['F1-Score']:.4f}")
print(f"Accuracy: {best['Accuracy']:.4f}")

# ---------- 7. Prediction Fields ----------

def get_patient_data():
    print("\nEnter Patient Details")

    patient = {}

    for col in X.columns:
        value = input(f"{col}: ").strip()

        if col in numeric_cols:
            try:
                patient[col] = float(value)
            except ValueError:
                print(f"Invalid number for {col}.")
                return None
        else:
            patient[col] = value

    return pd.DataFrame([patient], columns=X.columns)

# ---------- 8. Selection Menu ----------

while True:
    print("\n" + "=" * 60)
    print("PCOS EARLY DETECTION MENU")
    print("=" * 60)
    print("1. Predict PCOS Status")
    print("2. Compare Algorithms")
    print("3. View PCOS Status Distribution")
    print("4. Exit")

    choice = input("\nSelect number: ").strip()

    if choice == "1":
        print("\nSelect Algorithm:")
        print("1. Random Forest")
        print("2. Decision Tree")
        print("3. Both Algorithms")

        alg_choice = input("Select number: ").strip()

        if alg_choice == "1":
            selected = ["Random Forest"]
        elif alg_choice == "2":
            selected = ["Decision Tree"]
        elif alg_choice == "3":
            selected = list(trained_models.keys())
        else:
            print("Invalid algorithm selection.")
            continue

        patient_data = get_patient_data()

        if patient_data is None:
            continue

        print("\n" + "-" * 60)
        print("PCOS STATUS PREDICTION")
        print("-" * 60)

        for name in selected:
            prediction = trained_models[name].predict(
                patient_data
            )[0]

            print(f"{name} Prediction: {prediction}")

        print("\nNote: This is an experimental model prediction,")
        print("not a medical diagnosis.")

    elif choice == "2":
        print("\n===== ALGORITHM COMPARISON =====")
        print(comparison.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1-Score": "{:.4f}".format
            }
        ))
        print("\nBest Model:", best["Algorithm"])

    elif choice == "3":
        print("\n===== PCOS STATUS DISTRIBUTION =====")
        print(y.value_counts().to_string())

    elif choice == "4":
        print("\nThank you for using PCOS Early Detection System!")
        break

    else:
        print("Invalid choice. Select a number from 1 to 4.")
