const PORT_API_URL = "http://localhost:5000/api/portfolio/overview";

const token = localStorage.getItem("token");
console.log("Portfolio token:", token);

document.addEventListener("DOMContentLoaded", () => {
    loadPortfolioOverview();
});

async function loadPortfolioOverview() {

    try {

        const token = localStorage.getItem("token");

        console.log("Portfolio token:", token);

        const response = await fetch(PORT_API_URL, {
            method: "GET",
            headers: {
                "Authorization": `Bearer ${token}`,
                "Content-Type": "application/json"
            }
        });

        console.log("Status:", response.status);

        const text = await response.text();

        console.log("Raw Response:", text);

        const data = JSON.parse(text);

        console.log("Parsed Data:", data);

        renderPortfolioOverview(data);

    }
    catch(err){

        console.error("Portfolio Error:", err);

    }

}
function renderPortfolioOverview(data) {

    const container = document.getElementById("portfolio-overview");

    container.innerHTML = `
        <div class="portfolio-grid">

            <div class="portfolio-item">
                <h4>Total Stocks</h4>
                <span>${data.total_stocks ?? 0}</span>
            </div>

            <div class="portfolio-item buy">
                <h4>BUY</h4>
                <span>${data.buy_count ?? 0}</span>
            </div>

            <div class="portfolio-item hold">
                <h4>HOLD</h4>
                <span>${data.hold_count ?? 0}</span>
            </div>

            <div class="portfolio-item sell">
                <h4>SELL</h4>
                <span>${data.sell_count ?? 0}</span>
            </div>

            <div class="portfolio-item">
                <h4>Confidence</h4>
                <span>${data.average_confidence ?? 0}%</span>
            </div>

            <div class="portfolio-item">
                <h4>Average Price</h4>
                <span>₹${data.average_price ?? 0}</span>
            </div>

            <div class="portfolio-item">
                <h4>Sentiment</h4>
                <span>${data.overall_sentiment ?? "N/A"}</span>
            </div>

            <div class="portfolio-item">
                <h4>Last Updated</h4>
                <span>${formatDate(data.last_updated)}</span>
            </div>

        </div>
    `;
}

function formatDate(date) {

    if (!date) return "N/A";

    return new Date(date).toLocaleString();
}

async function getPortfolioOverview(req, res) {

    console.log("User:", req.user);

    const data = await portfolioService.getPortfolioOverview(req.user.id);

    console.log("Sending:", data);

    res.json(data);

}

window.loadPortfolioOverview = loadPortfolioOverview;