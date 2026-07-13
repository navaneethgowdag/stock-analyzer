const jwt = require("jsonwebtoken");

exports.verifyToken = (req, res, next) => {

    const authHeader = req.headers.authorization;

    if (!authHeader) {
        return res.status(401).json({
            message: "Unauthorized"
        });
    }

    // Remove "Bearer "
    const token = authHeader.startsWith("Bearer ")
        ? authHeader.split(" ")[1]
        : authHeader;

    jwt.verify(token, process.env.JWT_SECRET, (err, decoded) => {

        if (err) {
            console.log("JWT Error:", err);

            return res.status(401).json({
                message: "Invalid Token"
            });
        }

        req.user = decoded;

        next();

    });

};