const express = require("express");

const router = express.Router();

const {
    getMarketData
} = require("../controllers/marketController");

const {
    verifyToken
} = require("../middleware/authMiddleware");

router.get(
    "/",
    verifyToken,
    getMarketData
);

module.exports = router;