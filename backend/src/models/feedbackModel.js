const pool = require("../config/db");


// ==========================================
// Create Feedback
// ==========================================

exports.createFeedback = async ({
    userId,
    type,
    subject,
    message,
    rating
}) => {

    const query = `
        INSERT INTO user_feedback
        (
            user_id,
            type,
            subject,
            message,
            rating
        )
        VALUES ($1, $2, $3, $4, $5)
        RETURNING
            id,
            user_id,
            type,
            subject,
            message,
            rating,
            created_at
    `;

    const values = [
        userId,
        type,
        subject || null,
        message,
        rating || null
    ];

    const result = await pool.query(query, values);

    return result.rows[0];
};


// ==========================================
// Get User Feedback
// ==========================================

exports.getUserFeedback = async (userId) => {

    const query = `
        SELECT
            id,
            type,
            subject,
            message,
            rating,
            created_at
        FROM user_feedback
        WHERE user_id = $1
        ORDER BY created_at DESC
    `;

    const result = await pool.query(query, [userId]);

    return result.rows;
};