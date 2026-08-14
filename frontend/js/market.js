// ==========================================
// MARKET API
// ==========================================

const MARKET_API_URL =
    "https://stock-analyzer-backend-server.onrender.com";


// ==========================================
// Load Market Data
// ==========================================

async function loadMarketData() {

    const container =
        document.getElementById(
            "market-container"
        );

    if (!container) {
        return;
    }


    try {

        const token =
            localStorage.getItem("token");


        if (!token) {

            container.innerHTML = `
                <div class="market-empty">
                    Please login to view market data.
                </div>
            `;

            return;
        }


        const response = await fetch(
            `${MARKET_API_URL}/api/market`,
            {
                method: "GET",

                headers: {
                    "Authorization":
                        `Bearer ${token}`
                }
            }
        );


        if (!response.ok) {

            throw new Error(
                `Market API returned ${response.status}`
            );

        }


        const stocks =
            await response.json();


        renderMarketData(stocks);

    }
    catch (error) {

        console.error(
            "Market fetch error:",
            error
        );


        container.innerHTML = `
            <div class="market-empty">
                Unable to load market data.
            </div>
        `;

    }

}


// ==========================================
// Render Market Data
// ==========================================

function renderMarketData(stocks) {

    const container =
        document.getElementById(
            "market-container"
        );


    if (!stocks || !stocks.length) {

        container.innerHTML = `
            <div class="market-empty">
                Add stocks to your watchlist
                to see market data.
            </div>
        `;

        return;
    }


    container.innerHTML = "";


    stocks.forEach(stock => {

        const card =
            document.createElement("div");


        card.className =
            "market-stock-item";


        // ==================================
        // Direction
        // ==================================

        let directionClass =
            "market-neutral";

        let arrow = "—";


        if (stock.direction === "up") {

            directionClass =
                "market-up";

            arrow = "↑";

        }
        else if (
            stock.direction === "down"
        ) {

            directionClass =
                "market-down";

            arrow = "↓";

        }


        // ==================================
        // Price
        // ==================================

        const price =
            stock.currentPrice !== null
                ? `₹${Number(
                    stock.currentPrice
                ).toLocaleString(
                    "en-IN",
                    {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2
                    }
                )}`
                : "--";


        // ==================================
        // Change
        // ==================================

        let changeText = "--";

        if (
            stock.change !== null &&
            stock.changePercent !== null
        ) {

            const sign =
                stock.change > 0
                    ? "+"
                    : "";

            changeText =
                `${sign}${stock.change.toFixed(2)}
                (${sign}${stock.changePercent.toFixed(2)}%)`;

        }


        // ==================================
        // HTML
        // ==================================

        card.innerHTML = `

            <div class="market-stock-left">

                <div class="market-stock-symbol">

                    ${escapeMarketHTML(
                        stock.symbol
                    )}

                </div>

                <div class="market-company-name">

                    ${escapeMarketHTML(
                        stock.companyName || ""
                    )}

                </div>

            </div>


            <div class="market-stock-right">

                <div class="market-price">

                    ${price}

                </div>


                <div
                    class="market-change ${directionClass}"
                >

                    <span class="market-arrow">
                        ${arrow}
                    </span>

                    <span>
                        ${changeText}
                    </span>

                </div>

            </div>

        `;


        // ==================================
        // Click
        // ==================================

        card.addEventListener(
            "click",
            () => {

                showMarketStockDetails(
                    stock
                );

            }
        );


        container.appendChild(card);

    });

}


// ==========================================
// Escape HTML
// ==========================================

function escapeMarketHTML(value) {

    const div =
        document.createElement("div");

    div.textContent =
        value ?? "";

    return div.innerHTML;

}


// ==========================================
// Market Stock Details
// ==========================================

function showMarketStockDetails(stock) {

    const price =
        stock.currentPrice !== null
            ? `₹${Number(
                stock.currentPrice
            ).toLocaleString(
                "en-IN",
                {
                    minimumFractionDigits: 2
                }
            )}`
            : "--";


    const change =
        stock.change !== null
            ? stock.change.toFixed(2)
            : "--";


    const changePercent =
        stock.changePercent !== null
            ? stock.changePercent.toFixed(2)
            : "--";


    alert(
        `${stock.symbol}

Current Price: ${price}
Change: ${change}
Change %: ${changePercent}%`
    );

}


// ==========================================
// Auto Refresh
// ==========================================

let marketRefreshInterval;


function startMarketAutoRefresh() {

    clearInterval(
        marketRefreshInterval
    );


    marketRefreshInterval =
        setInterval(
            () => {

                loadMarketData();

            },
            30000
        );

}


// ==========================================
// Refresh After Watchlist Change
// ==========================================

window.refreshMarketData =
    function () {

        loadMarketData();

    };


// ==========================================
// DOM READY
// ==========================================

document.addEventListener(
    "DOMContentLoaded",
    () => {

        loadMarketData();

        startMarketAutoRefresh();

    }
);