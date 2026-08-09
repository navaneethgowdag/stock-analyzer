const pool = require("../config/db");


// ==========================================
// Get user's alerts
// ==========================================

async function getUserAlerts(userId) {

    const query = `
        SELECT
            a.id,
            a.ticker,
            a.alert_type,
            a.title,
            a.message,
            a.severity,
            a.value,
            a.reference_id,
            a.created_at

        FROM alerts a

        INNER JOIN watchlist w
            ON w.symbol = a.ticker

        WHERE w.user_id = $1

        ORDER BY a.created_at DESC

        LIMIT 20
    `;

    const result = await pool.query(
        query,
        [userId]
    );

    return result.rows;
}


// ==========================================
// Get latest alerts
// ==========================================

async function getLatestAlerts(
    userId,
    limit = 5
) {

    const query = `
        SELECT
            a.id,
            a.ticker,
            a.alert_type,
            a.title,
            a.message,
            a.severity,
            a.value,
            a.reference_id,
            a.created_at

        FROM alerts a

        INNER JOIN watchlist w
            ON w.symbol = a.ticker

        WHERE w.user_id = $1

        ORDER BY a.created_at DESC

        LIMIT $2
    `;

    const result = await pool.query(
        query,
        [userId, limit]
    );

    return result.rows;
}


module.exports = {
    getUserAlerts,
    getLatestAlerts
};