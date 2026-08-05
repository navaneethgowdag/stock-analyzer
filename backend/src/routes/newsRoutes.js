const express = require("express");

const router = express.Router();

const controller = require("../controllers/newsController");

const { verifyToken } = require("../middleware/authMiddleware");

router.get(
    "/",
    verifyToken,
    controller.getNews
);

module.exports = router;