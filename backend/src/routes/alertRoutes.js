const express = require("express");

const router = express.Router();

const alertController =
    require("../controllers/alertController");

const {
    verifyToken
} = require("../middleware/authMiddleware");


// ==========================================
// Get all alerts
// GET /api/alerts
// ==========================================

router.get(
    "/",
    verifyToken,
    alertController.getAlerts
);


// ==========================================
// Get latest alerts
// GET /api/alerts/latest
// ==========================================

router.get(
    "/latest",
    verifyToken,
    alertController.getLatestAlerts
);


module.exports = router;