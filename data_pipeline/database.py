import sqlite3
import pandas as pd
import os

DB_NAME = "zepto_books.db"
OUTPUT_DIR = "sql_outputs"


# ============================================================
# CREATE DATABASE
# ============================================================

def create_database():

    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON")

    conn.execute("DROP TABLE IF EXISTS books")
    conn.execute("DROP TABLE IF EXISTS categories")

    conn.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    conn.commit()
    conn.close()

    print("Database created successfully.")


# ============================================================
# INSERT DATA
# ============================================================

def insert_data(df):

    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON")

    # Insert categories
    categories = df["category"].drop_duplicates().tolist()

    for category in categories:
        conn.execute(
            "INSERT INTO categories (category_name) VALUES (?)",
            (category,)
        )

    # Insert books
    for _, row in df.iterrows():

        category_id = conn.execute(
            """
            SELECT category_id
            FROM categories
            WHERE category_name = ?
            """,
            (row["category"],)
        ).fetchone()[0]

        conn.execute(
            """
            INSERT INTO books
            (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(row["in_stock"]),
                category_id
            )
        )

    conn.commit()
    conn.close()

    print(f"{len(df)} books inserted into database.")


# ============================================================
# REQUIRED SQL QUERIES
# ============================================================

def run_queries():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    queries = {

        # 1. SELECT + WHERE
        "01_select_where": """
            SELECT title, rating, price_gbp
            FROM books
            WHERE rating >= 4;
        """,

        # 2. ORDER BY + LIMIT
        "02_order_by_limit": """
            SELECT title, price_inr, rating
            FROM books
            ORDER BY price_inr DESC, title ASC
            LIMIT 10;
        """,

        # 3. DISTINCT
        "03_distinct": """
            SELECT DISTINCT category_id
            FROM books
            ORDER BY category_id;
        """,

        # 4. BETWEEN
        "04_between": """
            SELECT title, price_gbp, rating
            FROM books
            WHERE price_gbp BETWEEN 20 AND 40;
        """,

        # 5. IN
        "05_in": """
            SELECT title, rating, category_id
            FROM books
            WHERE rating IN (4, 5);
        """,

        # 6. JOIN
        #
        # This is intentionally NOT limited.
        # The top 10 will be selected later,
        # identically in SQL and Pandas.
        #
        "06_join": """
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
        """
    }

    conn = sqlite3.connect(DB_NAME)

    for name, query in queries.items():

        result = pd.read_sql_query(
            query,
            conn
        )

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)
        print(result.head(10).to_string(index=False))

        # Save query
        with open(
            os.path.join(
                OUTPUT_DIR,
                name + ".sql"
            ),
            "w"
        ) as f:
            f.write(query.strip())

        # Save output
        result.to_csv(
            os.path.join(
                OUTPUT_DIR,
                name + ".csv"
            ),
            index=False
        )

    conn.close()


# ============================================================
# READ TWO SQL RESULTS USING pd.read_sql()
# ============================================================

def read_two_query_results():

    conn = sqlite3.connect(DB_NAME)

    result1 = pd.read_sql_query(
        """
        SELECT title, rating, price_gbp
        FROM books
        WHERE rating >= 4;
        """,
        conn
    )

    result2 = pd.read_sql_query(
        """
        SELECT title, price_inr, rating
        FROM books
        ORDER BY price_inr DESC, title ASC
        LIMIT 10;
        """,
        conn
    )

    conn.close()

    print("\n" + "=" * 60)
    print("pd.read_sql() RESULT 1")
    print("=" * 60)
    print(result1.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("pd.read_sql() RESULT 2")
    print("=" * 60)
    print(result2.to_string(index=False))

    result1.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "read_sql_result_1.csv"
        ),
        index=False
    )

    result2.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "read_sql_result_2.csv"
        ),
        index=False
    )

    return result1, result2


# ============================================================
# SQL JOIN VS PANDAS MERGE
# ============================================================

def compare_join_with_merge():

    conn = sqlite3.connect(DB_NAME)

    # --------------------------------------------------------
    # STEP 1
    # Get the COMPLETE SQL JOIN.
    # --------------------------------------------------------

    sql_join = pd.read_sql_query(
        """
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
        """,
        conn
    )

    # --------------------------------------------------------
    # STEP 2
    # Read BOTH original tables into memory.
    # --------------------------------------------------------

    books_df = pd.read_sql_query(
        """
        SELECT *
        FROM books
        """,
        conn
    )

    categories_df = pd.read_sql_query(
        """
        SELECT *
        FROM categories
        """,
        conn
    )

    conn.close()

    # --------------------------------------------------------
    # STEP 3
    # Reproduce JOIN ONLY with pd.merge()
    # --------------------------------------------------------

    pandas_join = pd.merge(
        books_df,
        categories_df,
        how="inner",
        on="category_id"
    )

    pandas_join = pandas_join[
        [
            "book_id",
            "title",
            "category_name",
            "rating",
            "price_gbp",
            "price_inr"
        ]
    ]

    # --------------------------------------------------------
    # STEP 4
    # APPLY IDENTICAL SORT TO BOTH
    # --------------------------------------------------------

    sql_join = sql_join.sort_values(
        by=["rating", "title"],
        ascending=[False, True],
        kind="merge sort"
    ).reset_index(drop=True)

    pandas_join = pandas_join.sort_values(
        by=["rating", "title"],
        ascending=[False, True],
        kind="merge sort"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # STEP 5
    # NOW take the same TOP 10 from both
    # --------------------------------------------------------

    sql_top10 = sql_join.head(10).copy()

    pandas_top10 = pandas_join.head(10).copy()

    # --------------------------------------------------------
    # STEP 6
    # Normalize values
    # --------------------------------------------------------

    for df in [sql_top10, pandas_top10]:

        df["book_id"] = df["book_id"].astype(int)

        df["title"] = df["title"].astype(str)

        df["category_name"] = (
            df["category_name"].astype(str)
        )

        df["rating"] = (
            df["rating"]
            .astype(int)
        )

        df["price_gbp"] = (
            df["price_gbp"]
            .astype(float)
            .round(3)
        )

        df["price_inr"] = (
            df["price_inr"]
            .astype(float)
            .round(3)
        )

    # --------------------------------------------------------
    # STEP 7
    # IMPORTANT:
    # Compare records after removing index.
    # --------------------------------------------------------

    sql_records = sql_top10.to_dict(
        orient="records"
    )

    pandas_records = pandas_top10.to_dict(
        orient="records"
    )

    equivalent = (
        sql_records == pandas_records
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    sql_top10.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "sql_join_result.csv"
        ),
        index=False
    )

    pandas_top10.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "pandas_merge_result.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SQL JOIN RESULT - TOP 10")
    print("=" * 60)

    print(
        sql_top10.to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("PANDAS MERGE RESULT - TOP 10")
    print("=" * 60)

    print(
        pandas_top10.to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("JOIN COMPARISON")
    print("=" * 60)

    print(
        "SQL JOIN == Pandas MERGE:",
        equivalent
    )

    if equivalent:

        print(
            "SUCCESS: Both approaches produced "
            "equivalent output."
        )

    else:

        print(
            "FAILED: SQL JOIN and Pandas MERGE differ."
        )

        # Show exactly which records differ
        print("\nSQL records:")
        print(sql_records)

        print("\nPandas records:")
        print(pandas_records)

    return equivalent


# ============================================================
# DATABASE SUMMARY
# ============================================================

def database_summary():

    conn = sqlite3.connect(DB_NAME)

    books = pd.read_sql_query(
        "SELECT COUNT(*) AS total_books FROM books",
        conn
    ).iloc[0, 0]

    categories = pd.read_sql_query(
        "SELECT COUNT(*) AS total_categories FROM categories",
        conn
    ).iloc[0, 0]

    conn.close()

    print("\n" + "=" * 60)
    print("DATABASE SUMMARY")
    print("=" * 60)

    print("Total books:", books)
    print("Total categories:", categories)

    return books, categories