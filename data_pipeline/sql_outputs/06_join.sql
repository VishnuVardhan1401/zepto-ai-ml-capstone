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