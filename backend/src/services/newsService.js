const newsModel = require("../models/newsModel");

exports.getNews = async (userId) => {

    const result = await newsModel.getNews(userId);

    return result.rows;

};