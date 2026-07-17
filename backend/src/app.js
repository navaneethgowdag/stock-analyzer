const express = require("express");
const cors = require("cors");

const authRoutes = require("./routes/authRoutes");

const app = express();

app.use(cors());
app.use(express.json());

console.log(authRoutes);

app.use("/api/auth", authRoutes);


app.get("/", (req, res) => {
    res.json({
        success: true,
        message: "AI Stock Monitor Backend Running 🚀"
    });
});

const portfolioRoutes = require("./routes/portfolioRoutes");
app.use("/api/portfolio", portfolioRoutes);
console.log("Portfolio routes registered");

const dashboardRoutes = require("./routes/dashboardRoutes");
app.use("/api/dashboard", dashboardRoutes);

const watchlistRoutes = require("./routes/watchlistRoutes");
app.use("/api/watchlist",watchlistRoutes);

module.exports = app;