const pool = require("../config/db");

exports.getMarketData = async (userId) => {

    const query = `
        SELECT
            w.id,
            w.symbol,
            w.company_name,

            p.ticker,
            p.current_price,
            p.previous_close,
            p.updated_at

        FROM watchlist w

        LEFT JOIN predictions p
            ON (
                UPPER(TRIM(p.ticker)) = UPPER(TRIM(w.symbol))
                OR
                UPPER(TRIM(p.symbol)) = UPPER(TRIM(w.symbol))
            )

        WHERE w.user_id = $1

        ORDER BY w.id ASC;
    `;

    const result = await pool.query(query, [userId]);

    return result.rows.map(stock => {

        const currentPrice =
            stock.current_price !== null
                ? Number(stock.current_price)
                : null;

        const previousClose =
            stock.previous_close !== null
                ? Number(stock.previous_close)
                : null;

        let change = null;
        let changePercent = null;
        let direction = "neutral";

        if (
            currentPrice !== null &&
            previousClose !== null &&
            previousClose !== 0
        ) {

            change =
                currentPrice - previousClose;

            changePercent =
                (change / previousClose) * 100;

            if (change > 0) {
                direction = "up";
            }
            else if (change < 0) {
                direction = "down";
            }
            else {
                direction = "neutral";
            }
        }

        return {
            id: stock.id,

            symbol: stock.symbol,

            companyName:
                stock.company_name || "",

            currentPrice,

            previousClose,

            change:
                change !== null
                    ? Number(change.toFixed(2))
                    : null,

            changePercent:
                changePercent !== null
                    ? Number(changePercent.toFixed(2))
                    : null,

            direction,

            updatedAt: stock.updated_at
        };
    });
};