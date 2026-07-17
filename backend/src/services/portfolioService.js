const portfolioModel = require("../models/portfolioModel");

exports.getPortfolioOverview = async(userId)=>{

    const result = await portfolioModel.getPortfolioOverview(userId);

    return result.rows[0];

};