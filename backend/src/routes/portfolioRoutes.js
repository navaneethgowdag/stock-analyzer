const express = require("express");

const router = express.Router();

const controller = require("../controllers/portfolioController");

const { verifyToken } = require("../middleware/authMiddleware");

router.get(
    "/overview",
    verifyToken,
    controller.getPortfolioOverview
);

module.exports = router;