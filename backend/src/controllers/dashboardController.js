const dashboardService = require("../services/dashboardService");

exports.getDashboard = async (req, res) => {

    try {

        const dashboard = await dashboardService.getDashboard(req.user.id);

        res.status(200).json(dashboard);

    }

    catch (error) {

        console.error(error);

        res.status(500).json({

            message: "Unable to load dashboard"

        });

    }

};