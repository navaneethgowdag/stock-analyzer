const pool = require("../config/db");

exports.getPortfolioOverview = async (userId) => {

    const query = `
        SELECT

            COUNT(*) AS total_stocks,

            COUNT(*) FILTER (
                WHERE p.recommendation='BUY'
            ) AS buy_count,

            COUNT(*) FILTER (
                WHERE p.recommendation='HOLD'
            ) AS hold_count,

            COUNT(*) FILTER (
                WHERE p.recommendation='SELL'
            ) AS sell_count,

            ROUND(AVG(p.prob_up)*100,2) AS average_confidence,

            ROUND(AVG(p.current_price),2) AS average_price,

            MODE() WITHIN GROUP (
                ORDER BY p.sentiment_label
            ) AS overall_sentiment,

            MAX(p.updated_at) AS last_updated

        FROM watchlist w

        INNER JOIN predictions p

        ON w.symbol = p.symbol

        WHERE w.user_id = $1;
    `;

    return await pool.query(query,[userId]);

};