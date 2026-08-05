const pool = require("../config/db");

exports.getPortfolioOverview = async (userId) => {

    const query = `
    SELECT

        COUNT(w.symbol) AS total_stocks,

        COUNT(*) FILTER (WHERE p.recommendation='BUY') AS buy_count,

        COUNT(*) FILTER (WHERE p.recommendation='HOLD') AS hold_count,

        COUNT(*) FILTER (WHERE p.recommendation='SELL') AS sell_count,

        COALESCE(ROUND(AVG(p.prob_up)*100,2),0) AS average_confidence,

        COALESCE(ROUND(AVG(p.current_price),2),0) AS average_price,

        COALESCE(
            MODE() WITHIN GROUP (ORDER BY p.sentiment_label),
            'N/A'
        ) AS overall_sentiment,

        MAX(p.updated_at) AS last_updated

    FROM watchlist w

    LEFT JOIN predictions p
    ON w.symbol = p.ticker

    WHERE w.user_id = $1;
    `;
    return await pool.query(query,[userId]);

};