# Zepto Data & AI Platform

An end-to-end Artificial Intelligence and Machine Learning project covering data collection, data engineering, exploratory data analysis, machine learning, and an AI-powered support assistant.

## Project Overview

This project implements a complete data and AI workflow through three modules:

1. **Module 1 — Data Pipeline**

   * Web scraping from Books to Scrape
   * Data cleaning and transformation
   * SQLite database creation
   * SQL querying
   * Pandas SQL-result analysis

2. **Module 2 — Analytics & Machine Learning**

   * Titanic dataset analysis
   * Exploratory Data Analysis
   * Data cleaning and missing-value handling
   * Outlier analysis
   * Statistical analysis and visualizations
   * Classification models
   * Imbalanced-data handling
   * Hyperparameter tuning
   * Fare regression

3. **Module 3 — Support Assistant**

   * Document ingestion
   * Text chunking and embeddings
   * ChromaDB vector database
   * Semantic retrieval
   * Structured prompt generation
   * LangGraph-based workflow
   * FastAPI interface
   * Offline mock LLM mode

---

# Project Structure

```text
zepto-ai-ml-capstone/
│
├── README.md
│
├── data_pipeline/
│   ├── scraper.py
│   ├── database.py
│   ├── run_pipeline.py
│   ├── zepto_books.db
│   └── sql_outputs/
│
├── analytics/
│   ├── titanic.csv
│   ├── eda.py
│   ├── modeling.py
│   └── README.md
│
└── support_assistant/
    ├── graph.py
    ├── ...
    └── README.md
```

---

# Module 1 — Data Pipeline

## Objective

The objective of Module 1 is to build an end-to-end data pipeline that collects book information from the web, cleans the data, stores it in a normalized SQLite database, and performs SQL and Pandas-based analysis.

## Data Source

The project uses:

**Books to Scrape**

```text
https://books.toscrape.com/
```

The pipeline collects books from three categories:

* Mystery
* Historical Fiction
* Sequential Art

The scraper follows pagination and collects at least 60 book records.

## Technologies Used

* Python
* Requests
* BeautifulSoup
* Pandas
* SQLite
* SQL

## Step 1 — Web Scraping

The scraper retrieves HTML pages using `requests` and parses them using BeautifulSoup.

The following information is collected:

* Title
* Price
* Star rating
* Availability
* Category

Pagination is handled so that multiple pages can be processed.

## Step 2 — Data Cleaning

The scraped data is converted into analysis-ready types.

### Price

The currency symbol is removed and the value is converted to a floating-point number.

```text
price_gbp
```

### Rating

The text ratings:

```text
One
Two
Three
Four
Five
```

are converted into:

```text
1
2
3
4
5
```

### Availability

Availability text is converted into a Boolean field:

```text
in_stock
```

### Currency Conversion

A fixed conversion rate is used:

```text
1 GBP = 105.50 INR
```

The INR price is calculated as:

```text
price_inr = price_gbp × 105.50
```

No external currency API is required.

## Step 3 — Database

The cleaned data is stored in SQLite.

The database uses a normalized structure containing:

* Categories
* Books

The category relationship is represented using primary and foreign keys.

This avoids unnecessary repetition of category information and provides a relational structure for SQL analysis.

## Step 4 — SQL Analysis

The pipeline demonstrates the following SQL operations:

* `SELECT`
* `WHERE`
* `ORDER BY`
* `LIMIT`
* `DISTINCT`
* `IN`
* `BETWEEN`
* `JOIN`

The SQL query results are saved in the `sql_outputs/` directory.

## Step 5 — Pandas Validation

SQL query results are loaded using:

```python
pd.read_sql()
```

The relational join is also reproduced using:

```python
pd.merge()
```

This demonstrates equivalent database and Pandas workflows.

## Running Module 1

From the project root:

```bash
python3 data_pipeline/run_pipeline.py
```

The pipeline performs scraping, cleaning, database insertion, and SQL analysis.

---

# Module 2 — Analytics & Machine Learning

## Objective

Module 2 performs complete exploratory data analysis and machine learning using the Titanic dataset.

## Data Source

The Titanic dataset is loaded using Seaborn:

```python
sns.load_dataset("titanic")
```

The dataset is immediately saved as:

```text
analytics/titanic.csv
```

This CSV is then used as the offline copy for subsequent analysis.

## Technologies Used

* Python
* Pandas
* NumPy
* Seaborn
* Matplotlib
* Scikit-learn
* Imbalanced-learn
* Joblib

## Step 1 — Data Inspection

The analysis includes:

* Dataset shape
* Data types
* Missing-value percentages
* Descriptive statistics
* Dataset information

## Step 2 — Missing-Value Handling

Missing values are handled according to their proportion and relevance.

The general approach is:

* Low missingness → row removal where appropriate
* Moderate missingness → imputation
* High missingness → dropping or encoding with justification

All preprocessing is performed without using test data to influence training transformations.

## Step 3 — Exploratory Data Analysis

The project analyzes:

* Age distribution
* Fare distribution
* Age outliers
* Fare outliers
* Fare mean
* Fare median
* Fare mode
* Fare skewness
* Survival by sex
* Survival by passenger class
* Survival by sex and passenger class

Box plots, histograms, bar charts, and correlation visualizations are used to understand the dataset.

## Step 4 — Correlation Analysis

Correlation is calculated for:

```text
survived
pclass
age
sibsp
parch
fare
```

A correlation heatmap is created and the strongest absolute correlations are identified.

## Step 5 — Outlier Analysis

The IQR method is used to identify outliers.

Z-scores are also calculated for:

* Age
* Fare

The analysis documents the number and characteristics of detected outliers.

## Step 6 — Classification

The target variable is:

```text
survived
```

The project evaluates multiple classification algorithms:

* Logistic Regression
* Decision Tree
* Random Forest

The preprocessing workflow uses Scikit-learn pipelines and includes:

* Missing-value imputation
* Categorical encoding
* Feature scaling where required
* Train/test separation

The train/test split is stratified to preserve the target-class distribution.

## Step 7 — Class Imbalance

The project compares:

1. Baseline model
2. `class_weight="balanced"`
3. SMOTE applied only to the training data

This allows the effect of imbalance-handling techniques to be evaluated without introducing test-data leakage.

## Step 8 — Random Forest Tuning

Random Forest hyperparameters are tuned using `GridSearchCV`.

The search includes:

```text
n_estimators
max_depth
max_features
```

The Random Forest also uses:

```text
oob_score=True
```

The resulting OOB score is reported as part of the model evaluation.

## Step 9 — Model Evaluation

Classification models are evaluated using metrics including:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

The results are organized into a model comparison table.

## Step 10 — Fare Regression

A regression task is also performed to predict passenger fare.

The model is evaluated using:

* MAE
* RMSE
* R²
* Adjusted R²

Residual analysis is performed to examine the relationship between prediction errors and predicted values, including potential heteroscedasticity.

## Step 11 — Model Persistence

The selected model is saved using Joblib.

The saved model is then reloaded and used to generate predictions to verify that model persistence works correctly.

## Running Module 2

From the project root:

```bash
cd analytics
python3 eda.py
python3 modeling.py
```

The analysis uses the committed:

```text
titanic.csv
```

as the offline dataset after the initial dataset creation.

---

# Module 3 — Support Assistant

## Objective

Module 3 implements an AI-powered support assistant using document retrieval, embeddings, ChromaDB, structured prompting, and a LangGraph workflow.

The assistant is designed to retrieve relevant information from a predefined support corpus and generate a response based on the retrieved context.

## Technologies Used

* Python
* LangGraph
* ChromaDB
* Embeddings
* FastAPI
* Pydantic
* Retrieval-Augmented Generation concepts

## Step 1 — Document Corpus

The support assistant uses a corpus of **8 documents** containing information that can be retrieved to answer support-related questions.

The documents are processed before being added to the vector database.

## Step 2 — Text Processing

Documents are split into smaller chunks to make semantic retrieval more effective.

Each chunk is converted into an embedding representation.

## Step 3 — ChromaDB

The embeddings are stored in ChromaDB.

The vector database allows the assistant to perform semantic similarity searches.

The acceptance criteria require all 8 corpus documents to be embedded and queryable.

## Step 4 — Retrieval

When a user submits a question:

1. The question is converted into an embedding.
2. ChromaDB searches for relevant document chunks.
3. The most relevant context is retrieved.
4. The retrieved context is passed to the response-generation workflow.

This provides a Retrieval-Augmented Generation style architecture.

## Step 5 — Structured Prompt

The project uses a structured prompt template containing:

1. Role/instruction
2. Context
3. User question
4. Response requirements
5. Output constraints

The prompt also includes:

* A negative constraint
* A few-shot example

The complete prompt structure is represented as actual prompt text rather than only being described conceptually.

## Step 6 — LangGraph Workflow

LangGraph is used to organize the assistant workflow.

The graph connects the major stages of the assistant, such as:

```text
User Query
     ↓
Retrieval
     ↓
Context
     ↓
Prompt Construction
     ↓
Response Generation
     ↓
Final Response
```

This provides a structured and extensible workflow.

## Step 7 — Offline Mock LLM

The project supports an offline mock LLM mode.

This allows the application to run without:

* Paid API services
* API keys
* External LLM network calls

The default mock mode is used for the baseline submission.

An optional real-LLM mode can use the structured prompt when configured.

## Step 8 — FastAPI

A FastAPI interface is provided for interacting with the support assistant.

The API accepts a user query and returns the generated assistant response.

The API can be used to test the assistant independently from the command-line workflow.

## Running Module 3

From the project root:

```bash
cd support_assistant
python3 graph.py
```

To run the FastAPI application, use the project's configured FastAPI/Uvicorn command, for example:

```bash
uvicorn graph:app --reload
```

The exact command depends on where the FastAPI application object is defined.

---

# Complete Project Workflow

The complete project can be understood as three connected engineering stages:

```text
                 ZEpto Data & AI Platform
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
   Data Pipeline      Analytics       Support Assistant
          │               │                │
          ▼               ▼                ▼
   Web Scraping       EDA + ML       Document Corpus
          │               │                │
          ▼               ▼                ▼
    Data Cleaning     Modeling        Embeddings
          │               │                │
          ▼               ▼                ▼
       SQLite          Evaluation       ChromaDB
          │                                │
          │                                ▼
          │                           Retrieval
          │                                │
          │                                ▼
          │                         Structured Prompt
          │                                │
          │                                ▼
          │                           LangGraph
          │                                │
          └───────────────┬────────────────┘
                          ▼
                    End-to-End AI/ML
                       Platform
```

---

# Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd zepto-ai-ml-capstone
```

## 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate it:

### macOS/Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

## 3. Install Dependencies

Install the dependencies required by the three modules.

For example:

```bash
pip install requests beautifulsoup4 pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib chromadb fastapi uvicorn langgraph
```

If individual modules contain their own `requirements.txt`, install those requirements as well.

---

# Running the Complete Project

## Module 1

From the project root:

```bash
python3 data_pipeline/run_pipeline.py
```

This runs:

```text
Scraping
   ↓
Cleaning
   ↓
SQLite Database
   ↓
SQL Queries
   ↓
Saved SQL Outputs
```

## Module 2

```bash
cd analytics
python3 eda.py
python3 modeling.py
cd ..
```

This performs:

```text
Titanic Dataset
      ↓
Data Cleaning
      ↓
EDA
      ↓
Visualization
      ↓
Classification
      ↓
Imbalance Handling
      ↓
Hyperparameter Tuning
      ↓
Regression
      ↓
Model Evaluation
```

## Module 3

```bash
cd support_assistant
python3 graph.py
```

For the API:

```bash
uvicorn graph:app --reload
```

---

# Key Design Decisions

## 1. No Paid Services Required

The project is designed to run without requiring paid APIs.

Module 3 includes an offline mock LLM mode so the support assistant can be demonstrated without an external LLM API key.

## 2. Fixed Currency Conversion

A fixed conversion rate is used for reproducibility:

```text
1 GBP = 105.50 INR
```

This avoids dependency on external currency APIs.

## 3. SQLite for Module 1

SQLite provides a lightweight relational database without requiring a separate database server.

It also allows the project to demonstrate:

* Primary keys
* Foreign keys
* Normalization
* SQL queries
* Relational joins

## 4. Offline Titanic Dataset

The Titanic dataset is saved to `titanic.csv` immediately after the initial load.

This makes subsequent analysis reproducible even without another dataset download.

## 5. Pipeline-Based Preprocessing

Machine-learning preprocessing is implemented using Scikit-learn pipelines and transformers.

This helps prevent data leakage because transformations are fitted only on training data.

## 6. Stratified Classification Split

The classification dataset uses stratification so that the relative distribution of survival classes is maintained between training and testing data.

## 7. SMOTE Only on Training Data

SMOTE is applied only to the training data.

The test set remains untouched so that evaluation represents performance on unseen data.

## 8. Vector Retrieval for the Support Assistant

ChromaDB is used to store embeddings and retrieve semantically relevant document chunks.

This allows the support assistant to ground its responses in the provided corpus.

---

# Reproducibility

The project is designed to be reproducible using:

* Fixed currency conversion
* Saved Titanic CSV
* SQLite database
* Saved SQL outputs
* Pipeline-based preprocessing
* Saved machine-learning model
* Offline mock LLM mode
* Local vector database


# Expected Outcomes

After completing the project:

### Module 1

A working data pipeline that:

* Scrapes book data
* Cleans and transforms the data
* Stores the data in SQLite
* Executes SQL queries
* Validates SQL results using Pandas

### Module 2

A complete analytics and ML workflow that:

* Explores the Titanic dataset
* Handles missing values and outliers
* Produces visualizations
* Trains multiple classification models
* Handles class imbalance
* Tunes a Random Forest
* Performs fare regression
* Evaluates and persists models

### Module 3

A working support assistant that:

* Processes 8 corpus documents
* Creates embeddings
* Stores them in ChromaDB
* Retrieves relevant context
* Uses a structured prompt
* Runs through a LangGraph workflow
* Supports offline mock LLM execution
* Provides a FastAPI interface

---

# Conclusion

This project demonstrates an end-to-end AI/ML engineering workflow, starting with raw web data and continuing through data engineering, analytics, machine learning, vector retrieval, and an AI support application.

The three modules collectively demonstrate practical skills in:

* Data acquisition
* Data cleaning
* Relational databases
* SQL
* Exploratory data analysis
* Statistical analysis
* Machine learning
* Model evaluation
* Imbalanced learning
* Hyperparameter tuning
* Model persistence
* Embeddings
* Vector databases
* Retrieval-Augmented Generation concepts
* LangGraph workflows
* FastAPI development

The project is designed to be reproducible, modular, and runnable locally without requiring paid external services.

