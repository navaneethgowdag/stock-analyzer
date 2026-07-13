const authService = require("../services/authService");

exports.register = async (req, res) => {
    try {

        const result = await authService.register(req.body);

        res.status(201).json(result);

    } catch (error) {

        console.error("LOGIN ERROR:", error);

        res.status(500).json({
            success: false,
            message: error.message || "Internal Server Error"
        });

    }
};

exports.login = async (req, res) => {
    try {

        const result = await authService.login(req.body);

        res.status(200).json(result);

    } catch (error) {

        console.error("LOGIN ERROR:", error);

        res.status(500).json({
            success: false,
            message: error.message || "Internal Server Error"
        });

    }
};