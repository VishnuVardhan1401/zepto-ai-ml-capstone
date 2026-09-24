from scraper import scrape_and_clean

from database import (
    create_database,
    insert_data,
    run_queries,
    read_two_query_results,
    compare_join_with_merge,
    database_summary
)


def main():

    print("=" * 60)
    print("MODULE 1 - DATA PIPELINE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. SCRAPE AND CLEAN DATA
    # ---------------------------------------------------------

    print("STARTING WEB SCRAPING")

    df = scrape_and_clean()

    print("\nScraping and cleaning completed.")
    print("Total cleaned books:", len(df))

    # ---------------------------------------------------------
    # 2. CREATE DATABASE
    # ---------------------------------------------------------

    print("\nCREATING DATABASE")

    create_database()

    # ---------------------------------------------------------
    # 3. INSERT DATA
    # ---------------------------------------------------------

    print("\nINSERTING DATA")

    insert_data(df)

    # ---------------------------------------------------------
    # 4. RUN REQUIRED SQL QUERIES
    # ---------------------------------------------------------

    print("\nRUNNING REQUIRED SQL QUERIES")

    run_queries()

    # ---------------------------------------------------------
    # 5. READ TWO SQL RESULTS USING pd.read_sql()
    # ---------------------------------------------------------

    print("\nREADING SQL RESULTS")

    read_two_query_results()

    # ---------------------------------------------------------
    # 6. COMPARE SQL JOIN WITH PANDAS MERGE
    # ---------------------------------------------------------

    print("\nCOMPARING SQL JOIN WITH PANDAS MERGE")

    equivalent = compare_join_with_merge()

    # ---------------------------------------------------------
    # 7. DATABASE SUMMARY
    # ---------------------------------------------------------

    database_summary()

    # ---------------------------------------------------------
    # 8. FINAL STATUS
    # ---------------------------------------------------------

    print("\n" + "=" * 60)

    if equivalent:
        print("MODULE 1 COMPLETED SUCCESSFULLY")
        print("SQL JOIN == Pandas MERGE: True")
    else:
        print("MODULE 1 COMPLETED WITH JOIN COMPARISON ISSUE")
        print("SQL JOIN == Pandas MERGE: False")

    print("=" * 60)


if __name__ == "__main__":
    main()