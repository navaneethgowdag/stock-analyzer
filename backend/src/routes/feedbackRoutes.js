const express = require("express");

const router = express.Router();

const feedbackController =
    require("../controllers/feedbackController");

const {
    verifyToken
} = require("../middleware/authMiddleware");


// ==========================================
// Submit Feedback / Suggestion
// ==========================================

router.post(
    "/",
    verifyToken,
    feedbackController.submitFeedback
);


// ==========================================
// Get Current User Feedback
// ==========================================

router.get(
    "/",
    verifyToken,
    feedbackController.getUserFeedback
);


module.exports = router;