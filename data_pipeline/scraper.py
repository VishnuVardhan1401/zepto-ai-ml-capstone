import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.parse import urljoin

CATEGORIES = {
    "Mystery":
        "https://books.toscrape.com/catalogue/category/books/mystery_3/index.html",

    "Historical Fiction":
        "https://books.toscrape.com/catalogue/category/books/historical-fiction_4/index.html",

    "Poetry":
        "https://books.toscrape.com/catalogue/category/books/poetry_23/index.html"
}

GBP_TO_INR = 105.50


def scrape_books():

    books = []

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for category, url in CATEGORIES.items():

        while url:

            response = requests.get(
                url,
                headers=headers,
                timeout=10
            )

            response.raise_for_status()

            soup = BeautifulSoup(
                response.content,
                "html.parser"
            )

            for item in soup.select("article.product_pod"):

                title = item.h3.a.get(
                    "title", ""
                ).strip()

                price = item.select_one(
                    ".price_color"
                ).get_text(strip=True)

                rating_tag = item.select_one(
                    ".star-rating"
                )

                rating = rating_tag.get(
                    "class", ["", "Unknown"]
                )[1]

                availability = item.select_one(
                    ".availability"
                ).get_text(" ", strip=True)

                books.append({
                    "title": title,
                    "price": price,
                    "star_rating": rating,
                    "availability": availability,
                    "category": category
                })

            next_button = soup.select_one(
                "li.next a"
            )

            if next_button:
                url = urljoin(
                    url,
                    next_button["href"]
                )
            else:
                url = None

    return pd.DataFrame(books)


def clean_price(value):

    match = re.search(
        r"\d+(?:\.\d+)?",
        str(value)
    )

    if match:
        return float(match.group())

    return None


def clean_data(df):

    df = df.copy()

    # Price
    df["price_gbp"] = df["price"].apply(
        clean_price
    )

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    # Rating
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    df["rating"] = df["star_rating"].map(
        rating_map
    )

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    # Availability
    # Parse availability into boolean
    df["in_stock"] = df["availability"].apply(
    lambda x: "In stock" in str(x)
)
    # Handle invalid values
    df["price_gbp"] = df["price_gbp"].fillna(
        df["price_gbp"].median()
    )

    df["rating"] = df["rating"].fillna(
        df["rating"].median()
    )

    df["rating"] = (
        df["rating"]
        .round()
        .astype(int)
    )

    # Remove invalid required rows
    df = df.dropna(
        subset=[
            "title",
            "category",
            "price_gbp",
            "rating"
        ]
    )

    # GBP to INR
    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    )

    return df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category"
        ]
    ]


def scrape_and_clean():

    print("=" * 60)
    print("STARTING WEB SCRAPING")
    print("=" * 60)

    raw_df = scrape_books()

    print("Raw books scraped:", len(raw_df))

    df = clean_data(raw_df)

    print("Clean books available:", len(df))
    print("Categories:", df["category"].nunique())

    if len(df) < 60:
        raise ValueError("At least 60 books are required.")

    if df["category"].nunique() < 3:
        raise ValueError("At least 3 categories are required.")

    print("\nCleaned data:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    return df
