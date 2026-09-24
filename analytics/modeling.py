import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# LOAD THE OFFLINE TITANIC CSV
# ============================================================

print("\n" + "=" * 70)
print("MODULE 2 - PREDICTIVE MODELING")
print("=" * 70)

df = pd.read_csv("/Users/vishnuvardhan/Desktop/zepto-ai-ml-capstone/analytics/titanic.csv")

print("\nDataset loaded from titanic.csv")

print("Shape:", df.shape)


# ============================================================
# TASK 7 - STRATIFIED TRAIN/TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TASK 7 - STRATIFIED TRAIN/TEST SPLIT")
print("=" * 70)


# Classification target
y = df["survived"].astype(int)

# Features
X = df.drop(
    columns=["survived"]
)


print("\nCLASS BALANCE")

class_counts = y.value_counts().sort_index()

print(class_counts)


print("\nCLASS BALANCE (%)")

class_percentages = (
    y.value_counts(
        normalize=True
    )
    .sort_index()
    * 100
)

print(
    class_percentages.round(2)
)


# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining set:", X_train.shape)

print("Testing set:", X_test.shape)


print("\nTraining class proportions:")

print(
    y_train.value_counts(
        normalize=True
    ).round(3)
)


print("\nTesting class proportions:")

print(
    y_test.value_counts(
        normalize=True
    ).round(3)
)


print("""
JUSTIFICATION:
Stratification preserves approximately the same proportion of
survived and not-survived observations in both training and test
sets. This is important because the target classes are imbalanced,
and an unstratified split could create an unrepresentative test set.
""")


# ============================================================
# TASK 8 - TRAIN-ONLY PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("TASK 8 - PREPROCESSING")
print("=" * 70)

classification_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
    "sex",
    "embarked"
]

classification_features = [
    column
    for column in classification_features
    if column in X_train.columns
]


X_train_cls = X_train[
    classification_features
].copy()

X_test_cls = X_test[
    classification_features
].copy()


numeric_features = [
    column
    for column in [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare"
    ]
    if column in classification_features
]


categorical_features = [
    column
    for column in [
        "sex",
        "embarked"
    ]
    if column in classification_features
]


print(
    "\nNumeric features:",
    numeric_features
)

print(
    "Categorical features:",
    categorical_features
)


# Numeric:
# median imputation + StandardScaler

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "scaler",
        StandardScaler()
    )
])

try:

    one_hot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

except TypeError:

    one_hot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )


# Categorical:
# most-frequent imputation + one-hot encoding

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (
        "encoder",
        one_hot_encoder
    )
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),

    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


print("""
PREPROCESSING DECISION:
Numeric features use median imputation followed by StandardScaler.
Categorical features use most-frequent imputation followed by
one-hot encoding. All preprocessing is placed inside the Pipeline,
so it is fitted only on the training split and then applied to the
test split without refitting.
""")


# ============================================================
# TASK 9 - TRAIN THREE CLASSIFIERS
# ============================================================

print("\n" + "=" * 70)
print("TASK 9 - THREE CLASSIFIERS")
print("=" * 70)


logistic_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


decision_tree_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )
    )
])


random_forest_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    )
])


models = {
    "Logistic Regression":
        logistic_pipeline,

    "Decision Tree":
        decision_tree_pipeline,

    "Random Forest":
        random_forest_pipeline
}


# Train all three
for name, model in models.items():

    model.fit(
        X_train_cls,
        y_train
    )

    print(
        f"{name}: training completed"
    )


# ------------------------------------------------------------
# DECISION TREE VISUALIZATION
# ------------------------------------------------------------

tree_model = decision_tree_pipeline.named_steps[
    "classifier"
]

tree_preprocessor = decision_tree_pipeline.named_steps[
    "preprocessor"
]


tree_feature_names = (
    tree_preprocessor
    .get_feature_names_out()
)


plt.figure(
    figsize=(24, 12)
)

plot_tree(
    tree_model,
    feature_names=tree_feature_names,
    class_names=[
        "Not Survived",
        "Survived"
    ],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title(
    "Decision Tree - Titanic Survival"
)

plt.tight_layout()

plt.show()


# ============================================================
# TASK 10 - EVALUATE ALL THREE CLASSIFIERS
# ============================================================

print("\n" + "=" * 70)
print("TASK 10 - CLASSIFIER EVALUATION")
print("=" * 70)


classification_results = []

roc_results = {}


for name, model in models.items():

    # Predictions
    y_pred = model.predict(
        X_test_cls
    )

    # Probability for ROC-AUC
    y_probability = model.predict_proba(
        X_test_cls
    )[:, 1]


    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )


    tn, fp, fn, tp = cm.ravel()


    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_probability
    )


    classification_results.append({
        "Model": name,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc
    })


    # ROC
    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability
    )

    roc_results[name] = (
        fpr,
        tpr,
        auc
    )


classification_table = pd.DataFrame(
    classification_results
)


print("\nCLASSIFICATION COMPARISON TABLE")

print(
    classification_table.round(4).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# CONFUSION MATRICES
# ------------------------------------------------------------

for name, model in models.items():

    y_pred = model.predict(
        X_test_cls
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    plt.figure(
        figsize=(5, 4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=[
            "Not Survived",
            "Survived"
        ],
        yticklabels=[
            "Not Survived",
            "Survived"
        ]
    )

    plt.title(
        f"{name} - Confusion Matrix"
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------
# ROC CURVES
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)


for name, values in roc_results.items():

    fpr, tpr, auc = values

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


plt.title(
    "ROC Curves - Titanic Classifiers"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# TASK 11 - CLASS IMBALANCE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("TASK 11 - CLASS IMBALANCE")
print("=" * 70)


print("\nORIGINAL CLASS BALANCE")

print(
    y.value_counts()
)


print("\nORIGINAL CLASS BALANCE (%)")

print(
    (
        y.value_counts(
            normalize=True
        )
        * 100
    ).round(2)
)


# ------------------------------------------------------------
# BASELINE
# ------------------------------------------------------------

baseline_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


baseline_pipeline.fit(
    X_train_cls,
    y_train
)


baseline_prediction = baseline_pipeline.predict(
    X_test_cls
)


# ------------------------------------------------------------
# CLASS WEIGHT BALANCED
# ------------------------------------------------------------

balanced_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])


balanced_pipeline.fit(
    X_train_cls,
    y_train
)


balanced_prediction = balanced_pipeline.predict(
    X_test_cls
)


# ------------------------------------------------------------
# SMOTE
# ------------------------------------------------------------

smote_pipeline = ImbPipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "smote",
        SMOTE(
            random_state=42
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


smote_pipeline.fit(
    X_train_cls,
    y_train
)


smote_prediction = smote_pipeline.predict(
    X_test_cls
)


# ------------------------------------------------------------
# COMPARE THREE STRATEGIES
# ------------------------------------------------------------

imbalance_results = []


for strategy, prediction in [
    (
        "Baseline",
        baseline_prediction
    ),

    (
        "Class Weight Balanced",
        balanced_prediction
    ),

    (
        "SMOTE",
        smote_prediction
    )
]:

    imbalance_results.append({
        "Strategy": strategy,

        "Precision":
            precision_score(
                y_test,
                prediction,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                prediction,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_test,
                prediction,
                zero_division=0
            )
    })


imbalance_table = pd.DataFrame(
    imbalance_results
)


print("\nIMBALANCE STRATEGY COMPARISON")

print(
    imbalance_table.round(4).to_string(
        index=False
    )
)


best_imbalance_row = imbalance_table.loc[
    imbalance_table["F1"].idxmax()
]


print(
    f"""
IMBALANCE CONCLUSION:
The best strategy by F1 score is
{best_imbalance_row["Strategy"]}, with an F1 score of
{best_imbalance_row["F1"]:.4f}. F1 balances precision and recall,
so it provides a useful overall comparison when the target classes
are imbalanced. The individual precision and recall values should
also be considered when selecting the final strategy.
"""
)


# ============================================================
# TASK 12 - RANDOM FOREST GRIDSEARCHCV + OOB
# ============================================================

print("\n" + "=" * 70)
print("TASK 12 - RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)

rf_tuning_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        RandomForestClassifier(
            random_state=42,
            oob_score=True
        )
    )
])


param_grid = {
    "classifier__n_estimators": [
        100,
        200
    ],

    "classifier__max_depth": [
        None,
        5,
        10
    ],

    "classifier__max_features": [
        "sqrt",
        "log2"
    ]
}


grid_search = GridSearchCV(
    estimator=rf_tuning_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)


print("\nRunning GridSearchCV...")

grid_search.fit(
    X_train_cls,
    y_train
)


best_rf_pipeline = (
    grid_search.best_estimator_
)


best_rf_estimator = (
    best_rf_pipeline
    .named_steps["classifier"]
)


print("\nBEST PARAMETERS")

print(
    grid_search.best_params_
)


print(
    "\nBEST CROSS-VALIDATION F1:",
    round(
        grid_search.best_score_,
        4
    )
)


print(
    "\nRANDOM FOREST OOB SCORE:",
    round(
        best_rf_estimator.oob_score_,
        4
    )
)


# ============================================================
# TASK 13 - MULTIVARIATE LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("TASK 13 - FARE REGRESSION")
print("=" * 70)


# Target
y_regression = df["fare"].copy()


X_regression = df.drop(
    columns=["fare"]
).copy()

regression_numeric = (
    X_regression
    .select_dtypes(
        include=["number"]
    )
    .columns
    .tolist()
)


regression_categorical = (
    X_regression
    .select_dtypes(
        include=["object", "category", "bool"]
    )
    .columns
    .tolist()
)


print(
    "\nRegression numeric features:",
    regression_numeric
)

print(
    "Regression categorical features:",
    regression_categorical
)


# Numeric regression preprocessing
regression_numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "scaler",
        StandardScaler()
    )
])


# Categorical regression preprocessing
try:

    regression_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

except TypeError:

    regression_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )


regression_categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (
        "encoder",
        regression_encoder
    )
])


# Regression ColumnTransformer
regression_preprocessor = ColumnTransformer([
    (
        "numeric",
        regression_numeric_pipeline,
        regression_numeric
    ),

    (
        "categorical",
        regression_categorical_pipeline,
        regression_categorical
    )
])


# Complete regression pipeline
regression_pipeline = Pipeline([
    (
        "preprocessor",
        regression_preprocessor
    ),

    (
        "regressor",
        LinearRegression()
    )
])


# Regression split
X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_regression,
        y_regression,
        test_size=0.20,
        random_state=42
    )
)


# Fit only on regression training data
regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)


# Predict test data
y_reg_pred = regression_pipeline.predict(
    X_reg_test
)


# ------------------------------------------------------------
# REGRESSION METRICS
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)


rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)


r2 = r2_score(
    y_reg_test,
    y_reg_pred
)

number_of_predictors = len(
    regression_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


number_of_observations = len(
    y_reg_test
)


if (
    number_of_observations
    - number_of_predictors
    - 1
    > 0
):

    adjusted_r2 = (
        1
        -
        (
            (1 - r2)
            *
            (number_of_observations - 1)
            /
            (
                number_of_observations
                - number_of_predictors
                - 1
            )
        )
    )

else:

    adjusted_r2 = np.nan


print("\nREGRESSION RESULTS")

print(
    f"MAE: {mae:.4f}"
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"R²: {r2:.4f}"
)

print(
    f"Adjusted R²: {adjusted_r2:.4f}"
)


# ------------------------------------------------------------
# RESIDUAL PLOT
# ------------------------------------------------------------

residuals = (
    y_reg_test
    - y_reg_pred
)


plt.figure(
    figsize=(8, 5)
)


sns.scatterplot(
    x=y_reg_pred,
    y=residuals
)


plt.axhline(
    0,
    linestyle="--"
)


plt.title(
    "Residual Plot - Fare Regression"
)

plt.xlabel(
    "Predicted Fare"
)

plt.ylabel(
    "Residual"
)


plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# HETEROSKEDASTICITY CHECK
# ------------------------------------------------------------

residual_analysis = pd.DataFrame({
    "predicted_fare": y_reg_pred,

    "absolute_residual":
        np.abs(residuals)
})


# Divide predicted values into four groups
residual_analysis["prediction_group"] = pd.qcut(
    residual_analysis["predicted_fare"],
    q=4,
    duplicates="drop"
)


average_absolute_residual = (
    residual_analysis
    .groupby(
        "prediction_group",
        observed=False
    )["absolute_residual"]
    .mean()
)


print(
    "\nAVERAGE ABSOLUTE RESIDUAL BY PREDICTED-FARE GROUP"
)

print(
    average_absolute_residual.round(4)
)


minimum_spread = (
    average_absolute_residual.min()
)

maximum_spread = (
    average_absolute_residual.max()
)


spread_ratio = (
    maximum_spread
    /
    max(
        minimum_spread,
        1e-9
    )
)


if spread_ratio > 2:

    heteroscedasticity_conclusion = (
        "The residual plot shows evidence of "
        "heteroscedasticity because the residual "
        "spread changes substantially across "
        "predicted fare values."
    )

else:

    heteroscedasticity_conclusion = (
        "The residual plot does not show strong "
        "evidence of heteroscedasticity because "
        "the residual spread is reasonably similar "
        "across predicted fare values."
    )


print(
    "\nHETEROSKEDASTICITY CONCLUSION:"
)

print(
    heteroscedasticity_conclusion
)


# ============================================================
# TASK 14 - MODEL COMPARISON TABLE + RECOMMENDATION
# ============================================================

print("\n" + "=" * 70)
print("TASK 14 - FINAL MODEL COMPARISON")
print("=" * 70)


# ------------------------------------------------------------
# Classification metrics
# ------------------------------------------------------------

classification_summary = classification_table[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AUC"
    ]
].copy()


# ------------------------------------------------------------
# Regression metrics
# ------------------------------------------------------------

regression_summary = pd.DataFrame([
    {
        "Model":
            "Multivariate Linear Regression",

        "MAE":
            mae,

        "RMSE":
            rmse,

        "R2":
            r2,

        "Adjusted R2":
            adjusted_r2
    }
])


print("\nCLASSIFICATION METRICS")

print(
    classification_summary.round(4).to_string(
        index=False
    )
)


print("\nREGRESSION METRICS")

print(
    regression_summary.round(4).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# COMBINED MODEL COMPARISON TABLE
# ------------------------------------------------------------

combined_rows = []


for _, row in classification_summary.iterrows():

    combined_rows.append({
        "Model Type":
            "Classification",

        "Model":
            row["Model"],

        "Accuracy":
            row["Accuracy"],

        "Precision":
            row["Precision"],

        "Recall":
            row["Recall"],

        "F1":
            row["F1"],

        "AUC":
            row["AUC"],

        "MAE":
            np.nan,

        "RMSE":
            np.nan,

        "R2":
            np.nan,

        "Adjusted R2":
            np.nan
    })


combined_rows.append({
    "Model Type":
        "Regression",

    "Model":
        "Multivariate Linear Regression",

    "Accuracy":
        np.nan,

    "Precision":
        np.nan,

    "Recall":
        np.nan,

    "F1":
        np.nan,

    "AUC":
        np.nan,

    "MAE":
        mae,

    "RMSE":
        rmse,

    "R2":
        r2,

    "Adjusted R2":
        adjusted_r2
})


model_comparison_table = pd.DataFrame(
    combined_rows
)


print("\nCOMPLETE MODEL COMPARISON TABLE")

print(
    model_comparison_table.round(4).to_string(
        index=False
    )
)


print("""
METRIC GROUPING:
Accuracy, Precision, Recall, F1 and AUC belong to the classification
models. MAE, RMSE, R² and Adjusted R² belong to the regression model.
These metric groups measure different tasks and therefore their numeric
values must not be interpreted as being directly comparable.
""")


# ------------------------------------------------------------
# FINAL CLASSIFIER RECOMMENDATION
# ------------------------------------------------------------

best_classifier_row = classification_summary.loc[
    classification_summary["F1"].idxmax()
]


best_classifier_name = (
    best_classifier_row["Model"]
)


best_accuracy = (
    best_classifier_row["Accuracy"]
)

best_precision = (
    best_classifier_row["Precision"]
)

best_recall = (
    best_classifier_row["Recall"]
)

best_f1 = (
    best_classifier_row["F1"]
)

best_auc = (
    best_classifier_row["AUC"]
)


print("""
FINAL WRITTEN RECOMMENDATION:
""")


print(
    f"""
I recommend deploying the {best_classifier_name} classifier because
it achieved the highest F1 score among the three required classifiers.
Its accuracy was {best_accuracy:.4f}, precision was {best_precision:.4f},
recall was {best_recall:.4f}, F1 score was {best_f1:.4f}, and ROC-AUC
was {best_auc:.4f}.

The F1 score provides a balanced view of precision and recall, while
ROC-AUC measures the model's ability to distinguish survivors from
non-survivors across classification thresholds. The final choice
should therefore consider both the F1 score and AUC rather than
accuracy alone.

The regression model is evaluated separately because predicting fare
is a continuous regression task and its MAE, RMSE, R² and Adjusted R²
metrics are not directly comparable with classification metrics.
"""
)


# ============================================================
# TASK 15 - SAVE AND RELOAD COMPLETE BEST PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("TASK 15 - SAVE AND RELOAD COMPLETE PIPELINE")
print("=" * 70)


# ------------------------------------------------------------
# Select the best required classifier based on F1
# ------------------------------------------------------------

best_pipeline = models[
    best_classifier_name
]


# Refit the selected complete pipeline on training data
best_pipeline.fit(
    X_train_cls,
    y_train
)


# ------------------------------------------------------------
# Save complete pipeline
# ------------------------------------------------------------

joblib.dump(
    best_pipeline,
    "best_pipeline.joblib"
)


print(
    "\nSaved complete pipeline:"
)

print(
    "best_pipeline.joblib"
)


# ------------------------------------------------------------
# Reload the complete pipeline
# ------------------------------------------------------------

loaded_pipeline = joblib.load(
    "best_pipeline.joblib"
)


print(
    "\nPipeline successfully reloaded."
)


# ------------------------------------------------------------
# RAW INPUT TEST
# ------------------------------------------------------------

raw_input = pd.DataFrame([
    {
        "pclass": 3,
        "age": 25,
        "sibsp": 0,
        "parch": 0,
        "fare": 10.0,
        "sex": "male",
        "embarked": "S"
    }
])


raw_input = raw_input[
    classification_features
]


raw_prediction = loaded_pipeline.predict(
    raw_input
)


raw_probability = (
    loaded_pipeline
    .predict_proba(raw_input)[:, 1]
)


print("\nRAW INPUT")

print(
    raw_input.to_string(
        index=False
    )
)


print("\nRELOADED PIPELINE PREDICTION")

if raw_prediction[0] == 1:

    print("Prediction: Survived")

else:

    print("Prediction: Not Survived")


print(
    f"Survival probability: "
    f"{raw_probability[0]:.4f}"
)


print("""
TASK 15 VERIFICATION:
The saved artifact contains the preprocessing steps and classifier
together as one scikit-learn Pipeline. The reloaded artifact accepts
raw numerical and categorical data without manual preprocessing and
successfully produces a prediction.
""")