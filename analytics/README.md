Module 2 — Analytics
Titanic EDA and Predictive Modeling

This module performs exploratory data analysis, data cleaning, classification modeling, class-imbalance analysis, hyperparameter tuning, and a regression side-task using the Titanic dataset.

The work is divided into:

Part A — Profiling, cleaning, and the data story
Part B — Predictive modeling

The module contains:

eda.py — Tasks 1–6
modeling.py — Tasks 7–15
titanic.csv — offline fallback dataset
best_pipeline.joblib — saved complete best-performing classification pipeline
README.md — documentation and interpretations

The Titanic dataset was loaded using Seaborn's Titanic dataset loader.

The dataset was profiled using:

df.info()
df.describe()
df.shape

The percentage of missing values was calculated for every column containing missing values.

Immediately after loading, the original DataFrame was saved as titanic.csv inside the analytics module. This provides an offline fallback so that the dataset can still be loaded with pd.read_csv("titanic.csv") if the Seaborn dataset cannot be accessed through the internet during grading.

The dataset is loaded only once. All subsequent EDA and modeling work is based on the same dataset or its saved CSV fallback.

Missing-value profile

The main missing-value observations are:

Age: approximately 19.87% missing
Embarked: less than 5% missing
Embark_town: less than 5% missing
Deck: approximately 77% missing

The exact percentages measured during execution are reported by the analysis.

#Task 2 — Missing-Value Handling

Missing values were handled according to the required threshold rule:

Missing-value percentage	Decision
Less than 5%	Drop affected rows
5%–30%	Impute missing values
More than 30%	Drop the column or encode missing as its own category, with justification
Age — 19.87%

Approximately 19.87% of the age values are missing.

Because the missing percentage falls between 5% and 30%, the missing age values are imputed. Median imputation is used because age is a numerical variable and the median is less affected by extreme observations than the mean.

Embarked — below 5%

The missing percentage for embarked is below 5%.

According to the required threshold rule, the affected rows are dropped rather than imputed.

Embark_town — below 5%

The missing percentage for embark_town is also below 5%.

The affected rows are therefore dropped according to the same threshold rule.

Deck — approximately 77%

Approximately 77% of the deck observations are missing.

This is substantially above the range where normal imputation would be reliable. Therefore, deck is dropped rather than attempting to impute such a large proportion of the column.

This decision avoids introducing a large amount of artificial information into the dataset.

Cleaning summary

The cleaning strategy therefore follows the specified threshold rather than applying one generic missing-value treatment to every column.

The exact missing percentage is measured before cleaning, and the corresponding decision is documented for each affected column.

#Task 3 — Univariate Analysis

Univariate analysis was performed on age and fare.

For both variables, the following visualizations were produced:

Histogram
Box plot
Age — Distribution

The age histogram shows the distribution of passenger ages.

The majority of passengers are concentrated in the younger and middle-age ranges, while comparatively fewer passengers are in the older age ranges.

Age — Outlier Analysis

Age outliers were identified using the required IQR rule:

IQR = Q3 − Q1

Lower bound = Q1 − 1.5 × IQR

Upper bound = Q3 + 1.5 × IQR

Any age observation outside these bounds is classified as an outlier.

The exact number of age outliers is calculated and reported by the analysis.

Fare — Distribution

The fare histogram shows a concentration of passengers at lower fare values with a smaller number of passengers paying substantially higher fares.

The distribution is therefore right-skewed.

This is supported by the ordering of the central-tendency measures, where the mean is higher than the median, with the mode representing the most frequently occurring fare value.

Fare — Outlier Analysis

The fare box plot shows several observations outside the IQR limits.

These are treated as statistical outliers for EDA purposes rather than automatically being removed, because high fares may represent genuine differences in passenger class and ticket pricing.

The exact number of fare outliers is reported using the required IQR rule.

Fare Statistics

The following statistics were calculated:

Mean
Median
Mode

The mean is higher than the median, which supports the conclusion that the fare distribution is right-skewed.

The mode identifies the most frequently occurring fare value, while the median provides a more robust measure of the typical fare because of the high-fare observations.

#Task 4 — Bivariate Analysis

Bivariate analysis was used to investigate survival relationships.

Survival rates were calculated using the binary survived variable, where:

0 = did not survive
1 = survived

The survival rate is therefore represented by the mean of the survived variable.

Boolean masking using combinations of & and | was used for the required breakdowns.

Survival by Sex

Survival rates were calculated separately for male and female passengers.

Female passengers had a substantially higher survival rate than male passengers.

This indicates that sex was a strong factor associated with survival in the Titanic dataset.

Survival by Passenger Class

Survival rates were calculated separately for:

First class
Second class
Third class

The analysis shows that survival rates differed substantially between passenger classes, with passengers in higher classes generally having better survival outcomes.

This indicates that passenger class was another important factor associated with survival.

Survival by Sex and Passenger Class

Survival rates were also calculated jointly by:

Sex
Passenger class

This combined analysis provides a more detailed view of survival than either variable alone.

The results show that the relationship between sex and survival also differs across passenger classes. Therefore, survival is better understood by considering multiple passenger characteristics together.

Correlation Analysis

The correlation matrix was deliberately restricted to exactly these six columns:

survived
pclass
age
sibsp
parch
fare

The boolean columns:

adult_male
alone

were excluded.

These columns are derived or redundant flags and are not treated as independent measured features for this correlation analysis.

The resulting 6 × 6 correlation matrix was visualized using a Seaborn heatmap.

Strongest Correlations

All off-diagonal correlation coefficients were ranked according to their absolute values.

The two feature pairs with the largest absolute off-diagonal correlation coefficients were selected as the two strongest correlations.

The analysis reports these two pairs and their correlation values.

The strongest relationships should be interpreted as associations rather than causal relationships, because correlation does not establish causation.

#Task 5 — Multivariate Data Story

At least four distinct charts were produced to create a coherent story about who was more likely to survive and why.

Each chart has a written interpretation, because a chart without interpretation does not satisfy the requirement.

Chart 1 — Survival Rate by Sex

This chart compares survival rates between male and female passengers.

The survival difference is substantial, with female passengers generally having a much higher survival rate. This supports the conclusion that sex was one of the strongest factors associated with Titanic survival.

Chart 2 — Survival Rate by Passenger Class

This chart compares survival rates across the three passenger classes.

Passengers in first class generally had higher survival rates than those in lower classes, while third-class passengers had the lowest survival outcomes. This suggests that passenger class and the socioeconomic position associated with it were important factors in survival.

Chart 3 — Survival Rate by Sex and Passenger Class

This chart combines sex and passenger class to show their relationship with survival.

The results demonstrate that survival differences between males and females persist across passenger classes, while class also affects the survival rate within sex groups. This provides a stronger multivariate explanation than considering sex or class independently.

Chart 4 — Fare Distribution by Survival

This chart compares fares for survivors and non-survivors.

Survivors generally show higher fare values than non-survivors. Because fare is related to passenger class, this provides additional evidence that socioeconomic position was associated with survival.

Overall Data Story

The EDA suggests that Titanic survival was not determined by a single characteristic.

Sex and passenger class show particularly strong relationships with survival, while age, fare, family-related variables, and other passenger characteristics provide additional information.

These findings motivate the use of multivariable machine-learning models rather than relying on one feature alone.

#Task 6 — Z-Score Standardization Sanity Check

As an exploratory check, age and fare were standardized using the z-score transformation:

z = (x − mean) / standard deviation

The means and standard deviations were compared before and after standardization.

Before transformation, the variables were measured on their original scales.

After transformation, the resulting columns have approximately:

Mean = 0
Standard deviation = 1

This confirms that the z-score transformation was applied correctly.

This standardization is used only as an EDA sanity check.

It is not used as the preprocessing step for the classification models. The actual modeling pipeline performs its own scaling after the train/test split so that preprocessing is fitted only on training data.

#Part B — Predictive Modeling
#Task 7 — Stratified Train/Test Split

The classification target is:

survived

The data is divided into training and testing sets using a stratified split.

Stratification is important because the target contains two classes:

Not survived
Survived

The class proportions observed in the original dataset should be preserved approximately in both the training and testing sets.

This provides a more representative test set and makes model evaluation more reliable.

The train/test split is performed before the modeling preprocessing.

Task 8 — Train-Only Preprocessing

The modeling preprocessing is handled separately from the EDA cleaning process.

The classification preprocessing includes:

Numerical features

The numerical features used by the classifier are:

pclass
age
sibsp
parch
fare

Missing numerical values are handled using imputation, followed by StandardScaler.

Categorical features

The categorical features are:

sex
embarked

Missing categorical values are imputed and the categorical variables are encoded using one-hot encoding.

Unknown categories are handled safely so that new raw observations can still be processed by the saved pipeline.

Data leakage prevention

All preprocessing operations are fitted only on the training split.

The test set is never used to fit:

Imputers
Encoders
Scalers

The fitted preprocessing steps are then applied to the test data in transform-only mode.

A ColumnTransformer and scikit-learn Pipeline are used to enforce this separation structurally.

This prevents information from the test set from leaking into the training process.

#Task 9 — Classification Models

Three classification models were trained using the same stratified train/test split:

Logistic Regression
Decision Tree
Random Forest
Logistic Regression

Logistic Regression provides a simple linear classification baseline.

It establishes a reference point against which the more complex tree-based models can be compared.

Decision Tree

The Decision Tree is capable of learning non-linear decision boundaries and feature interactions.

The fitted Decision Tree is visualized using plot_tree, including:

Feature names
Class names
Tree decision structure
Random Forest

Random Forest combines multiple decision trees.

This can provide better generalization and more stable predictions than a single Decision Tree.

All three models are evaluated on the same test set to ensure a fair comparison.

#Task 10 — Classification Evaluation

Each classifier was evaluated using the required metrics:

Accuracy
Precision
Recall
F1 score
ROC-AUC

A confusion matrix and ROC curve were also produced for each model.

Accuracy

Accuracy represents the proportion of all test observations that were classified correctly.

Precision

Precision measures how many observations predicted as survivors were actually survivors.

A high precision means fewer false-positive survivor predictions.

Recall

Recall measures how many actual survivors were successfully identified by the model.

A high recall means fewer actual survivors are missed.

F1 Score

F1 score combines precision and recall into one measure.

It is useful when both precision and recall are important.

ROC-AUC

ROC-AUC measures the model's ability to distinguish between the two survival classes across classification thresholds.

A higher AUC indicates better class discrimination.

Confusion Matrix

The confusion matrix shows:

True negatives
False positives
False negatives
True positives

These results provide more detail than accuracy alone.

#Task 11 — Class-Imbalance Comparison

The survival class distribution was first examined to determine whether the target classes were balanced.

Three approaches were then compared using the same classification model:

Baseline

The original training data was used without explicit imbalance handling.

Class Weight Balanced

The classifier was trained using balanced class weights.

This gives additional importance to the minority class during model training.

SMOTE

SMOTE was used to oversample the minority class by generating synthetic training examples.

SMOTE was applied only to the training data.

The test set was not oversampled.

This is essential to prevent data leakage and ensure that the evaluation represents performance on the original unseen data distribution.

The three approaches were compared using:

Precision
Recall
F1 score
Imbalance Conclusion

The best imbalance strategy is determined using the actual precision, recall, and F1 results produced by the model.

F1 score is particularly useful for this comparison because it balances precision and recall.

A strategy that improves recall but produces a large decrease in precision should not automatically be considered better. The final choice considers the overall trade-off between identifying survivors and avoiding incorrect survivor predictions.

#Task 12 — Random Forest Hyperparameter Tuning

Random Forest hyperparameters were optimized using GridSearchCV.

The required parameters searched were:

n_estimators
max_depth
max_features

The search evaluates different combinations using cross-validation and selects the best-performing combination according to the chosen scoring criterion.

The best parameter combination is reported in the modeling output.

Out-of-Bag Score

The Random Forest estimator was constructed with OOB evaluation enabled.

The corresponding OOB score is reported for the best Random Forest configuration.

The OOB score provides an additional estimate of model generalization based on observations that were not included in individual bootstrap samples.

The OOB score is reported separately from the test-set classification metrics.

#Task 13 — Multivariate Linear Regression

A separate regression task was performed to predict:

fare

using the other available features.

The fare column is removed from the predictor variables so that the model does not use the target itself as an input.

The regression model uses multiple available passenger characteristics to estimate fare.

Regression Evaluation

The regression model is evaluated using:

MAE
RMSE
R²
Adjusted R²
MAE

Mean Absolute Error measures the average absolute difference between actual and predicted fare.

Lower MAE indicates smaller average prediction error.

RMSE

Root Mean Squared Error penalizes larger prediction errors more strongly than MAE.

Lower RMSE indicates better predictive performance.

R²

R² measures how much of the variation in fare is explained by the regression model.

A higher R² indicates that more variation in fare is explained by the predictors.

Adjusted R²

Adjusted R² accounts for the number of predictors used by the model and penalizes unnecessary predictors.

It provides a more conservative measure than ordinary R² when multiple predictors are included.

Residual Analysis

A residual plot was produced to evaluate the regression model's errors.

Residuals are the difference between the actual and predicted fare values.

The residual plot was examined for patterns in the spread of residuals.

If residuals are randomly distributed around zero with approximately constant spread, the model does not show strong evidence of heteroscedasticity.

If the spread increases or decreases systematically as predicted fare increases, this indicates heteroscedasticity.

The final conclusion about heteroscedasticity is based on the residual plot generated by the analysis.

#Task 14 — Model Comparison

Classification and regression are treated as two separate metric groups because they solve different prediction problems.

Classification Model Comparison

The following table should contain the actual values produced by modeling.py:

Model	Accuracy	Precision	Recall	F1	AUC
Logistic Regression	[value]	[value]	[value]	[value]	[value]
Decision Tree	[value]	[value]	[value]	[value]	[value]
Random Forest	[value]	[value]	[value]	[value]	[value]
Regression Model Metrics

The regression metrics are reported separately:

Regression Model	MAE	RMSE	R²	Adjusted R²
Multivariate Linear Regression	[value]	[value]	[value]	[value]

The classification and regression metrics are not directly comparable.

Accuracy, precision, recall, F1, and AUC measure classification performance, whereas MAE, RMSE, R², and Adjusted R² measure continuous-value regression performance.

Therefore, the two metric groups are presented separately rather than treating all values as though they were on one common scale.

#Task 15 — Final Recommendation and Model Deployment
Final Classifier Recommendation

The final classifier is selected by comparing the three classification models using their:

Accuracy
Precision
Recall
F1 score
ROC-AUC

The model with the strongest overall performance should be selected while considering the trade-off between precision and recall.

Final Recommendation

Recommended model: [INSERT BEST MODEL FROM YOUR ACTUAL OUTPUT]

The recommended classifier achieved an accuracy of [value], precision of [value], recall of [value], F1 score of [value], and ROC-AUC of [value].

The model was selected because it provides the strongest overall balance of classification performance based on the evaluation results. F1 score is considered important because it balances precision and recall, while ROC-AUC provides an additional measure of the model's ability to distinguish survivors from non-survivors.

The final choice should also consider the imbalance comparison and Random Forest tuning results before deployment.

Complete Pipeline Saving

The best-performing classification model is saved together with its complete preprocessing pipeline.

The saved artifact contains the full sequence of:

Missing-value handling
Categorical encoding
Numerical scaling
Final classifier

The complete pipeline is saved as:

best_pipeline.joblib

Saving the complete pipeline instead of only the estimator ensures that raw future input can be passed directly to the model without manually reproducing the preprocessing steps.

Pipeline Reload and Prediction

The saved pipeline is reloaded from disk and tested on raw input.

The reload test confirms that:

The saved pipeline can be successfully loaded.
Raw, unprocessed input can be provided.
The preprocessing steps are automatically applied.
A survival prediction can be generated.
A survival probability can also be generated where supported.

This confirms that the saved artifact is usable as an end-to-end prediction pipeline.

Data Leakage Prevention

Data leakage was explicitly avoided throughout the modeling workflow.

The following practices were followed:

Train/test split is performed before model preprocessing.
The split is stratified using survived.
Imputers are fitted only on training data.
Encoders are fitted only on training data.
StandardScaler is fitted only on training data.
Test data is only transformed using preprocessing fitted on training data.
SMOTE is applied only to the training fold.
The test set remains untouched during SMOTE.
The EDA z-score standardization is separate from the modeling pipeline.
The final saved artifact contains preprocessing and the classifier together.

These steps ensure that information from the test data does not influence model training.

Reproducibility

A fixed random state is used where applicable to make the analysis reproducible.

The same train/test split is used for the three main classifiers so that their performance can be compared fairly.

The same cleaned dataset is used throughout the analysis and modeling workflow.

Final Findings

The exploratory analysis indicates that passenger survival was strongly associated with sex and passenger class.

Female passengers generally had considerably higher survival rates than male passengers, while passengers in higher classes generally had better survival outcomes.

The combined sex-and-class analysis provides a stronger explanation of survival than considering either variable independently.

Fare also provides useful information because higher fares are associated with passenger class and survivors generally show higher fare values.

The machine-learning stage evaluates Logistic Regression, Decision Tree, and Random Forest using multiple classification metrics rather than relying on accuracy alone.

Class imbalance is separately evaluated using a baseline model, balanced class weights, and SMOTE applied only to the training data.

Random Forest hyperparameters are optimized using GridSearchCV, and the required OOB score is reported.

The regression side-task predicts fare using the other available features and evaluates the model using MAE, RMSE, R², and Adjusted R², together with residual analysis for heteroscedasticity.

The final recommended classifier is selected based on the actual evaluation results, particularly the balance between precision, recall, F1, and AUC.

Module 2 Deliverables:
Contains the complete implementation of Tasks 1–6:

Dataset loading
Profiling
Missing-value percentages
Cleaning decisions
Age analysis
Fare analysis
IQR outlier detection
Survival analysis
Correlation analysis
Heatmap
Multivariate charts
Written chart interpretations
Z-score sanity check
modeling.py

Contains the complete implementation of Tasks 7–15:

Stratified train/test split
Train-only preprocessing
Logistic Regression
Decision Tree
Random Forest
Decision Tree visualization
Classification metrics
Confusion matrices
ROC curves and AUC
Class-imbalance comparison
Class-weight balancing
SMOTE
Random Forest GridSearchCV
OOB score
Multivariate fare regression
Regression metrics
Residual plot
Model comparison
Final recommendation
Complete pipeline saving
Pipeline reload and raw-input prediction
titanic.csv

Offline fallback copy of the originally loaded Titanic dataset.

best_pipeline.joblib

Saved complete preprocessing-and-classification pipeline for end-to-end prediction on raw input.

README.md