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
        document.getElementById("market-container");

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
                        `Bearer ${token}`,

                    "Content-Type":
                        "application/json"
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
// Get BUY / HOLD / SELL Suggestion
// ==========================================
//
// Priority:
//
// 1. Use suggestion coming from backend
// 2. Otherwise calculate using changePercent
//
// This keeps your frontend compatible with
// the existing API.
// ==========================================

function getMarketSuggestion(stock) {

    // ======================================
    // Backend suggestion
    // ======================================

    if (stock.suggestion) {

        const suggestion =
            String(stock.suggestion)
                .trim()
                .toUpperCase();


        if (
            suggestion === "BUY" ||
            suggestion === "HOLD" ||
            suggestion === "SELL"
        ) {

            return suggestion;

        }

    }


    // ======================================
    // Other possible backend field names
    // ======================================

    if (stock.recommendation) {

        const recommendation =
            String(stock.recommendation)
                .trim()
                .toUpperCase();


        if (
            recommendation === "BUY" ||
            recommendation === "HOLD" ||
            recommendation === "SELL"
        ) {

            return recommendation;

        }

    }


    if (stock.signal) {

        const signal =
            String(stock.signal)
                .trim()
                .toUpperCase();


        if (
            signal === "BUY" ||
            signal === "HOLD" ||
            signal === "SELL"
        ) {

            return signal;

        }

    }


    // ======================================
    // Fallback calculation
    // ======================================

    const changePercent =
        Number(stock.changePercent);


    if (
        stock.changePercent === null ||
        stock.changePercent === undefined ||
        Number.isNaN(changePercent)
    ) {

        return "HOLD";

    }


    /*
        Simple momentum-based fallback:

        >= +1%  → BUY
        <= -1%  → SELL
        between → HOLD

        IMPORTANT:
        This is only a simple UI fallback.
        If your Python/AI backend already
        calculates recommendations, return
        that value from the backend instead.
    */

    if (changePercent >= 1) {

        return "BUY";

    }


    if (changePercent <= -1) {

        return "SELL";

    }


    return "HOLD";

}


// ==========================================
// Suggestion CSS Class
// ==========================================

function getSuggestionClass(suggestion) {

    switch (suggestion) {

        case "BUY":
            return "market-suggestion-buy";

        case "SELL":
            return "market-suggestion-sell";

        case "HOLD":
        default:
            return "market-suggestion-hold";

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
        // Current Price
        // ==================================

        const price =
            stock.currentPrice !== null &&
            stock.currentPrice !== undefined

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
        // Previous Close
        // ==================================

        const previousClose =
            stock.previousClose !== null &&
            stock.previousClose !== undefined

                ? `₹${Number(
                    stock.previousClose
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
            stock.change !== undefined &&
            stock.changePercent !== null &&
            stock.changePercent !== undefined
        ) {

            const change =
                Number(stock.change);

            const changePercent =
                Number(stock.changePercent);


            const sign =
                change > 0
                    ? "+"
                    : "";


            changeText =
                `${sign}${change.toFixed(2)}
                (${sign}${changePercent.toFixed(2)}%)`;

        }


        // ==================================
        // BUY / HOLD / SELL
        // ==================================

        const suggestion =
            getMarketSuggestion(stock);


        const suggestionClass =
            getSuggestionClass(
                suggestion
            );


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


            <div class="market-stock-middle">

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


                <div class="market-previous-close">

                    Prev Close:
                    ${previousClose}

                </div>

            </div>


            <div class="market-stock-suggestion">

                <span
                    class="
                        market-suggestion
                        ${suggestionClass}
                    "
                >

                    ${suggestion}

                </span>

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

    // ======================================
    // Current Price
    // ======================================

    const price =
        stock.currentPrice !== null &&
        stock.currentPrice !== undefined

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


    // ======================================
    // Previous Close
    // ======================================

    const previousClose =
        stock.previousClose !== null &&
        stock.previousClose !== undefined

            ? `₹${Number(
                stock.previousClose
            ).toLocaleString(
                "en-IN",
                {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                }
            )}`

            : "--";


    // ======================================
    // Change
    // ======================================

    const change =
        stock.change !== null &&
        stock.change !== undefined

            ? Number(
                stock.change
            ).toFixed(2)

            : "--";


    // ======================================
    // Change %
    // ======================================

    const changePercent =
        stock.changePercent !== null &&
        stock.changePercent !== undefined

            ? Number(
                stock.changePercent
            ).toFixed(2)

            : "--";


    // ======================================
    // Suggestion
    // ======================================

    const suggestion =
        getMarketSuggestion(stock);


    // ======================================
    // Message
    // ======================================

    alert(
        `${stock.symbol}

Current Price: ${price}

Previous Close: ${previousClose}

Change: ${change}

Change %: ${changePercent}%

Suggestion: ${suggestion}`
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