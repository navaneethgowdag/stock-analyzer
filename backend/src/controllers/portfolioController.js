const portfolioService = require("../services/portfolioService");

exports.getPortfolioOverview = async(req,res)=>{

    try{

        const portfolio = await portfolioService.getPortfolioOverview(

            req.user.id

        );

        res.json(portfolio);

    }

    catch(err){

        console.error(err);

        res.status(500).json({

            message:"Unable to fetch portfolio overview"

        });

    }

};