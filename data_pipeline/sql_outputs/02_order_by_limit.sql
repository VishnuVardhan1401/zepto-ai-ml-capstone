SELECT title, price_inr, rating
            FROM books
            ORDER BY price_inr DESC, title ASC
            LIMIT 10;