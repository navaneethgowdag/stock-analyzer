const watchlistModel = require("../models/watchlistModel");

exports.addStock = async (userId,data)=>{

    const result = await watchlistModel.addStock(

        userId,

        data.symbol,

        data.companyName,

        data.exchange

    );

    return result.rows[0];

};

exports.getWatchlist = async(userId)=>{

    const result = await watchlistModel.getWatchlist(userId);

    return result.rows;

};

exports.deleteStock = async(userId,id)=>{

    const result = await watchlistModel.deleteStock(userId,id);

    return result.rows[0];

};