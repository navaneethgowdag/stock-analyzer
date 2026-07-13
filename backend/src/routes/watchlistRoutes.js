const express = require("express");

const router = express.Router();

const controller = require("../controllers/watchlistController");

const { verifyToken } = require("../middleware/authMiddleware");

router.post("/", verifyToken, controller.addStock);

router.get("/", verifyToken, controller.getWatchlist);

router.delete("/:id", verifyToken, controller.deleteStock);

module.exports = router;