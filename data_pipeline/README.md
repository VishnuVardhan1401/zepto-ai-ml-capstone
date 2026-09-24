# Data Pipeline
 # Capstone Project — Certificate Program in Artificial Intelligence and Machine Learning


## 1. Project Overview

This project is an end-to-end **AI/ML Engineering capstone** designed to demonstrate the complete workflow of collecting, processing, storing, analyzing, and building machine-learning solutions from real-world data.

The project is developed as a modular pipeline so that each stage has a clear responsibility and can be tested independently.

The overall workflow is:

```text
Data Source
    ↓
Data Collection / Scraping
    ↓
Data Cleaning & Transformation
    ↓
Database Storage
    ↓
SQL Querying
    ↓
Analytics / EDA
    ↓
Feature Engineering
    ↓
Machine Learning
    ↓
Model Evaluation
    ↓
Final Results / Insights
```

The implementation is divided into modules rather than placing the complete project in one script.

---

# 2. Project Objectives

The main objectives are to:

* Collect real-world data programmatically.
* Build a reliable data-cleaning pipeline.
* Convert raw scraped values into analysis-ready data types.
* Store structured data using a normalized relational database.
* Demonstrate SQL querying and relational operations.
* Validate SQL results using Pandas.
* Perform exploratory data analysis and derive meaningful insights.
* Prepare features for machine-learning tasks.
* Train and evaluate suitable ML models.
* Follow a reproducible and organized project structure.
* Produce outputs that can be inspected and submitted as project evidence.

---

# 3. Project Structure

The project follows a modular structure similar to:

```text
AIML-Capstone/
│
├── README.md
│
├── data_pipeline/
│   ├── scraper.py
│   ├── database.py
│   ├── run_pipeline.py
│   ├── zepto_books.db
│   └── sql_outputs/
│       ├── 01_select_where.sql
│       ├── 01_select_where.csv
│       ├── 02_order_by_limit.sql
│       ├── 02_order_by_limit.csv
│       ├── 03_distinct.sql
│       ├── 03_distinct.csv
│       ├── 04_between.sql
│       ├── 04_between.csv
│       ├── 05_in.sql
│       ├── 05_in.csv
│       ├── 06_join.sql
│       ├── 06_join.csv
│       ├── read_sql_result_1.csv
│       ├── read_sql_result_2.csv
│       ├── sql_join_result.csv
│       └── pandas_merge_result.csv
│
---

# 4. Technology Stack

The project uses Python as the primary programming language.

### Core technologies

* **Python**
* **Pandas**
* **SQLite**
* **Requests**
* **BeautifulSoup**
* **SQL**
* **Scikit-learn** for machine learning stages

Additional libraries can be added when required by later modules.

---

# 4. Module 1 — Data Pipeline

Module 1 is responsible for creating the project's initial structured dataset.

It performs:

```text
Scraping
   ↓
Cleaning
   ↓
Transformation
   ↓
SQLite Storage
   ↓
SQL Queries
   ↓
Pandas Validation
```

The data source used is **Books to Scrape**.

The scraper collects books from:

* Mystery
* Historical Fiction
* Poetry

The pipeline follows the category pages and their pagination links to collect the available books.

---

# 5. Module 1 — Raw Data

The scraper collects fields including:

| Field          | Description                |
| -------------- | -------------------------- |
| `title`        | Book title                 |
| `price`        | Original price text        |
| `star_rating`  | Text rating                |
| `availability` | Original availability text |
| `category`     | Book category              |

These values are initially scraped as web-page text and are not necessarily in the correct format for analysis.

---

# 6. Module 1 — Cleaning Decisions

Cleaning is performed before inserting the data into SQLite.

## Price

The pound symbol is removed from the original price.

Example:

```text
£48.35 → 48.35
```

The cleaned field is:

```text
price_gbp
```

and is stored as a numeric value.

## Rating

The text rating is converted to an integer:

```text
One   → 1
Two   → 2
Three → 3
Four  → 4
Five  → 5
```

The final field is:

```text
rating
```

## Availability

The availability text is converted into a Boolean:

```text
In stock (...) → True
Out of stock   → False
```

The final field is:

```text
in_stock
```

When stored in SQLite, Boolean values are represented internally as `1` and `0`.

## Currency conversion

The cleaned GBP value is converted to INR using the exchange rate configured in the pipeline:

```text
price_inr = price_gbp × 105.50
```

Example:

```text
48.35 × 105.50 = 5100.925
```

The final fields are:

```text
price_gbp
price_inr
rating
in_stock
```

---

# 7. Handling Invalid Data

The pipeline converts fields into appropriate numeric types and handles parsing failures rather than allowing invalid text values to enter the database.

For numeric fields, invalid values are represented as missing values during parsing and handled according to the cleaning logic.

Rows missing essential information are removed before database insertion.

This ensures that the database contains structured, usable data instead of raw website text.

---

# 8. Module 1 — Database Design

SQLite is used for persistent structured storage.

The database contains two normalized tables:

```text
categories
     │
     │ category_id
     ↓
books
```

## Categories

```sql
categories(
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE NOT NULL
)
```

## Books

```sql
books(
    book_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    price_gbp REAL,
    price_inr REAL,
    rating INTEGER,
    in_stock INTEGER,
    category_id INTEGER,
    FOREIGN KEY(category_id)
        REFERENCES categories(category_id)
)
```

The `category_id` creates the primary-key/foreign-key relationship between the tables.

This avoids repeatedly storing the category name for every book and provides a normalized relational design.

---

# 9. Module 1 — Data Insertion

Python's `sqlite3` library is used to insert the cleaned Pandas DataFrame into SQLite.

The insertion process is:

```text
Clean DataFrame
      ↓
Extract unique categories
      ↓
Insert categories
      ↓
Retrieve category_id
      ↓
Insert books
      ↓
SQLite database
```

Foreign-key enforcement is enabled using:

```python
conn.execute("PRAGMA foreign_keys = ON")
```

---

# 10. Module 1 — SQL Requirements

The pipeline executes SQL queries demonstrating the required operations:

* `SELECT`
* `WHERE`
* `ORDER BY`
* `LIMIT`
* `DISTINCT`
* `BETWEEN`
* `IN`
* `JOIN`

Six queries are implemented.

### Query 1

```sql
SELECT title, rating, price_gbp
FROM books
WHERE rating >= 4;
```

Demonstrates:

```text
SELECT + WHERE
```

### Query 2

```sql
SELECT title, price_inr, rating
FROM books
ORDER BY price_inr DESC, title ASC
LIMIT 10;
```

Demonstrates:

```text
ORDER BY + LIMIT
```

### Query 3

```sql
SELECT DISTINCT category_id
FROM books
ORDER BY category_id;
```

Demonstrates:

```text
DISTINCT
```

### Query 4

```sql
SELECT title, price_gbp, rating
FROM books
WHERE price_gbp BETWEEN 20 AND 40;
```

Demonstrates:

```text
BETWEEN
```

### Query 5

```sql
SELECT title, rating, category_id
FROM books
WHERE rating IN (4, 5);
```

Demonstrates:

```text
IN
```

### Query 6

```sql
SELECT
    b.book_id,
    b.title,
    c.category_name,
    b.rating,
    b.price_gbp,
    b.price_inr
FROM books b
JOIN categories c
    ON b.category_id = c.category_id
ORDER BY b.rating DESC, b.title ASC;
```

Demonstrates:

```text
JOIN
```

---

# 11. Saving Query Evidence

Each SQL query and its output are saved separately.

For example:

```text
01_select_where.sql
01_select_where.csv
```

This makes the SQL implementation reproducible and provides direct evidence of query execution.

The JOIN outputs are also saved separately:

```text
sql_join_result.csv
pandas_merge_result.csv
```

---

# 12. SQL JOIN and Pandas MERGE Validation

An important validation step is to reproduce the SQL JOIN using Pandas.

The two database tables are loaded into memory:

```python
books_df = pd.read_sql_query(
    "SELECT * FROM books",
    conn
)

categories_df = pd.read_sql_query(
    "SELECT * FROM categories",
    conn
)
```

The equivalent Pandas operation is:

```python
pandas_join = pd.merge(
    books_df,
    categories_df,
    how="inner",
    on="category_id"
)
```

The Pandas result is independently produced without using a SQL JOIN.

Both results are then sorted using:

```text
rating DESC
title ASC
```

and the same top-10 selection is applied.

This is important because multiple books can have the same rating. The title provides a deterministic tie-breaker.

The final comparison verifies whether:

```text
SQL JOIN == Pandas MERGE
```

The expected successful result is:

```text
SQL JOIN == Pandas MERGE: True
```

---

# 13. Module 1 Outputs

After running Module 1, the project produces:

```text
zepto_books.db
```

and the SQL evidence files inside:

```text
sql_outputs/
```

These outputs demonstrate:

* cleaned data storage,
* normalized relational database design,
* SQL querying,
* SQL JOIN,
* Pandas `read_sql_query()`,
* Pandas `merge()`,
* and JOIN equivalence validation.

---

# 14. Running the Data Pipeline

From the project root:

```bash
python run_pipeline.py
```

The complete Module 1 execution is:

```text
run_pipeline.py
       ↓
scraper.py
       ↓
Clean DataFrame
       ↓
database.py
       ↓
Create SQLite schema
       ↓
Insert data
       ↓
Run SQL queries
       ↓
Save query outputs
       ↓
Read SQL results with Pandas
       ↓
Pandas MERGE
       ↓
JOIN comparison
```

---

# 15. Reproducibility

The project should be executable from a clean environment.

A new user should be able to:

```text
1. Install dependencies
2. Run the pipeline
3. Generate the database
4. Generate SQL outputs
5. Inspect the saved results
```

The code should avoid relying on manually edited intermediate values.

---

# 16. Submission Structure

For final submission, the project should contain:

```text
AIML-Capstone/
│
├── README.md
├── requirements.txt
│
├── data_pipeline/
│   ├── scraper.py
│   ├── database.py
│   ├── run_pipeline.py
│   ├── zepto_books.db
│   └── sql_outputs/

---