const db = require("../config/db");

exports.getDashboard = async (userId) => {

    const result = await db.query(

        `
        SELECT
            id,
            full_name,
            email
        FROM users
        WHERE id = $1
        `,

        [userId]

    );

    const user = result.rows[0];

    return {

        user,

        watchlistCount: 0,

        marketStatus: "Loading...",

        lastUpdated: new Date().toLocaleString(),

        news: [],

        ai: {

            status: "Waiting"

        }

    };

};