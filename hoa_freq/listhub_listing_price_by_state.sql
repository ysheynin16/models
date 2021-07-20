SELECT
    state_or_province AS state,
    CAST(MEDIAN(CAST (list_price AS numeric)) AS INT) median_price
FROM
    listhub.listing_raw
WHERE
    CAST(
        listing_date AS DATE
    ) > '2020-01-01'
GROUP BY
    1;
