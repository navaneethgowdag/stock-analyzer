const watchlistService = require("../services/watchlistService");

exports.addStock = async(req,res)=>{

    try{

        const stock = await watchlistService.addStock(

            req.user.id,

            req.body

        );

        res.status(201).json(stock);

    }

    catch(err){

        console.error(err);

        res.status(500).json({

            message:"Unable to add stock"

        });

    }

};

exports.getWatchlist = async(req,res)=>{

    try{

        const stocks = await watchlistService.getWatchlist(

            req.user.id

        );

        res.json(stocks);

    }

    catch(err){

        res.status(500).json({

            message:"Unable to fetch watchlist"

        });

    }

};

exports.deleteStock = async(req,res)=>{

    try{

        await watchlistService.deleteStock(

            req.user.id,

            req.params.id

        );

        res.json({

            message:"Deleted"

        });

    }

    catch(err){

        res.status(500).json({

            message:"Delete failed"

        });

    }

};