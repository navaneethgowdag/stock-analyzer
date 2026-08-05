const pool = require("../config/db");

exports.getNews = async (userId) => {

    const query = `
        SELECT
            n.id,
            n.ticker,
            n.headline,
            n.label,
            n.score,
            n.confidence,
            n.created_at

        FROM news_sentiment n

        INNER JOIN watchlist w
            ON n.ticker = w.symbol

        WHERE w.user_id = $1

        ORDER BY n.created_at DESC;
    `;

    return pool.query(query, [userId]);

};