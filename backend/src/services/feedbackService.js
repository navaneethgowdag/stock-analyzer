const feedbackModel = require("../models/feedbackModel");


// ==========================================
// Submit Feedback
// ==========================================

exports.submitFeedback = async (userId, data) => {

    const {
        type,
        subject,
        message,
        rating
    } = data;


    // Validate type

    if (!type) {

        throw new Error("Feedback type is required.");

    }


    if (
        type !== "feedback" &&
        type !== "suggestion"
    ) {

        throw new Error("Invalid feedback type.");

    }


    // Validate message

    if (!message || !message.trim()) {

        throw new Error("Message is required.");

    }


    // Validate rating

    if (
        rating !== undefined &&
        rating !== null &&
        (
            Number(rating) < 1 ||
            Number(rating) > 5
        )
    ) {

        throw new Error("Rating must be between 1 and 5.");

    }


    return await feedbackModel.createFeedback({

        userId,

        type,

        subject:
            subject
                ? subject.trim()
                : null,

        message:
            message.trim(),

        rating:
            rating
                ? Number(rating)
                : null

    });

};


// ==========================================
// Get User Feedback
// ==========================================

exports.getUserFeedback = async (userId) => {

    return await feedbackModel.getUserFeedback(userId);

};