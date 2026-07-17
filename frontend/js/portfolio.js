const PORT_API_URL = "http://localhost:5000/api/portfolio/overview";

document.addEventListener("DOMContentLoaded", () => {
    loadPortfolioOverview();
});

async function loadPortfolioOverview() {

    try {
        const token = localStorage.getItem("token");

        if (!token) return;

        const response = await fetch(PORT_API_URL, {
            method: "GET",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
            }
        });


        if (!response.ok) {
            throw new Error("Failed to fetch portfolio overview");
        }

        const data = await response.json();

        renderPortfolioOverview(data);
    

    } catch (error) {
        console.error("Portfolio Overview Error:", error);

        document.getElementById("portfolio-overview").innerHTML = `
            <p class="widget-message">
                Unable to load portfolio overview.
            </p>
        `;
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