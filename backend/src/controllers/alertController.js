const alertService = require("../services/alertService");


// ==========================================
// Get all alerts
// ==========================================

exports.getAlerts = async (req, res) => {

    try {

        const userId = req.user.id;

        const alerts =
            await alertService.getUserAlerts(userId);

        res.status(200).json({
            success: true,
            alerts
        });

    } catch (error) {

        console.error(
            "Get alerts error:",
            error
        );

        res.status(500).json({
            success: false,
            message: "Failed to fetch alerts"
        });

    }

};


// ==========================================
// Get latest alerts
// ==========================================

exports.getLatestAlerts = async (req, res) => {

    try {

        const userId = req.user.id;

        const limit =
            Number(req.query.limit) || 5;

        const alerts =
            await alertService.getLatestAlerts(
                userId,
                limit
            );

        res.status(200).json({
            success: true,
            alerts
        });

    } catch (error) {

        console.error(
            "Get latest alerts error:",
            error
        );

        res.status(500).json({
            success: false,
            message: "Failed to fetch latest alerts"
        });

    }

};