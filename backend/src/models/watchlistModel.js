const db = require("../config/db");

exports.addStock = async (userId, symbol, companyName, exchange) => {

    const query = `
        INSERT INTO watchlist
        (user_id, symbol, company_name, exchange)
        VALUES ($1,$2,$3,$4)
        RETURNING *;
    `;

    return await db.query(query, [
        userId,
        symbol,
        companyName,
        exchange
    ]);

};

exports.getWatchlist = async (userId) => {

    const query = `
        SELECT *
        FROM watchlist
        WHERE user_id=$1
        ORDER BY created_at DESC;
    `;

    return await db.query(query,[userId]);

};

exports.deleteStock = async (userId,id)=>{

    return await db.query(

        `DELETE FROM watchlist
         WHERE id=$1
         AND user_id=$2
         RETURNING *;`,

        [id,userId]

    );

};