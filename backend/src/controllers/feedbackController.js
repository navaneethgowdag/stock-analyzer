const feedbackService = require("../services/feedbackService");


// ==========================================
// Submit Feedback
// ==========================================

exports.submitFeedback = async (req, res) => {

    try {

        const userId = req.user.id;

        const feedback =
            await feedbackService.submitFeedback(
                userId,
                req.body
            );


        return res.status(201).json({

            success: true,

            message:
                "Thank you! Your feedback has been submitted.",

            feedback

        });

    }

    catch (error) {

        console.error(
            "Feedback Error:",
            error.message
        );


        return res.status(400).json({

            success: false,

            message:
                error.message ||
                "Unable to submit feedback."

        });

    }

};


// ==========================================
// Get User Feedback
// ==========================================

exports.getUserFeedback = async (req, res) => {

    try {

        const userId = req.user.id;

        const feedback =
            await feedbackService.getUserFeedback(userId);


        return res.json({

            success: true,

            feedback

        });

    }

    catch (error) {

        console.error(
            "Get Feedback Error:",
            error.message
        );


        return res.status(500).json({

            success: false,

            message:
                "Unable to fetch feedback."

        });

    }

};