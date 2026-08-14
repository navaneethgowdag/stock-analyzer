const PORT_API_URL = "https://stock-analyzer-backend-server.onrender.com/api/portfolio/overview";


// ==========================================
// Load Portfolio When Page Loads
// ==========================================

document.addEventListener("DOMContentLoaded", () => {
    loadPortfolioOverview();
});


// ==========================================
// Load Portfolio Overview
// ==========================================

async function loadPortfolioOverview() {

    try {

        // Get JWT token
        const token = localStorage.getItem("token");

        // No token
        if (!token) {

            console.error("Portfolio: No authentication token found.");

            return;
        }


        const response = await fetch(PORT_API_URL, {

            method: "GET",

            headers: {

                "Authorization": `Bearer ${token}`,

                "Content-Type": "application/json"

            }

        });


        // Handle expired/invalid token
        if (response.status === 401) {

            console.warn("Portfolio: Token expired or invalid.");

            localStorage.removeItem("token");

            window.location.href = "/login.html";

            return;
        }


        // Handle other server errors
        if (!response.ok) {

            const errorData = await response.json().catch(() => ({}));

            throw new Error(
                errorData.message || `HTTP Error ${response.status}`
            );
        }


        // IMPORTANT:
        // Fetch JSON directly
        const data = await response.json();

        renderPortfolioOverview(data);

    }

    catch (err) {

        console.error("Portfolio Error:", err);

    }

}


// ==========================================
// Render Portfolio Overview
// ==========================================

function renderPortfolioOverview(data) {

    const container = document.getElementById("portfolio-overview");


    if (!container) {

        console.error(
            "Portfolio: #portfolio-overview element not found."
        );

        return;
    }


    container.innerHTML = `

        <div class="portfolio-grid">

            <div class="portfolio-item">

                <h4>Total Stocks</h4>

                <span>
                    ${data.total_stocks ?? 0}
                </span>

            </div>


            <div class="portfolio-item buy">

                <h4>BUY</h4>

                <span>
                    ${data.buy_count ?? 0}
                </span>

            </div>


            <div class="portfolio-item hold">

                <h4>HOLD</h4>

                <span>
                    ${data.hold_count ?? 0}
                </span>

            </div>


            <div class="portfolio-item sell">

                <h4>SELL</h4>

                <span>
                    ${data.sell_count ?? 0}
                </span>

            </div>


            <div class="portfolio-item">

                <h4>Confidence</h4>

                <span>
                    ${data.average_confidence ?? 0}%
                </span>

            </div>


            <div class="portfolio-item">

                <h4>Average Price</h4>

                <span>
                    ₹${data.average_price ?? 0}
                </span>

            </div>


            <div class="portfolio-item">

                <h4>Sentiment</h4>

                <span>
                    ${data.overall_sentiment ?? "N/A"}
                </span>

            </div>


            <div class="portfolio-item">

                <h4>Last Updated</h4>

                <span>
                    ${formatDate(data.last_updated)}
                </span>

            </div>

        </div>

    `;
}


// ==========================================
// Format Date
// ==========================================

function formatDate(date) {

    if (!date) {

        return "N/A";

    }

    return new Date(date).toLocaleString();

}


// ==========================================
// Make Function Globally Available
// ==========================================

window.loadPortfolioOverview = loadPortfolioOverview;