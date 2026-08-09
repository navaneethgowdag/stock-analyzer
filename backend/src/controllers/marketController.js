const marketService = require("../services/marketService");

exports.getMarketData = async (req, res) => {

    try {

        const userId = req.user.id;

        const marketData =
            await marketService.getMarketData(userId);

        res.status(200).json(marketData);

    }
    catch (error) {

        console.error(
            "Market Controller Error:",
            error
        );

        res.status(500).json({
            message: "Unable to fetch market data"
        });
    }
};