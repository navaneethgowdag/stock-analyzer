const rateLimit = require("express-rate-limit");


// ==========================================
// LOGIN RATE LIMITER
// ==========================================

const loginLimiter = rateLimit({

    windowMs: 15 * 60 * 1000,

    max: 10,

    message: {
        message: "Too many login attempts. Please try again later."
    },

    standardHeaders: true,

    legacyHeaders: false

});


// ==========================================
// GENERAL API RATE LIMITER
// ==========================================

const apiLimiter = rateLimit({

    windowMs: 15 * 60 * 1000,

    max: 300,

    message: {
        message: "Too many requests. Please try again later."
    },

    standardHeaders: true,

    legacyHeaders: false

});


module.exports = {
    loginLimiter,
    apiLimiter
};