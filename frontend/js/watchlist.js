// ==========================================
// Watchlist API
// ==========================================

const API_URL = "http://localhost:5000";

// ==========================================
// Add Stock
// ==========================================

async function addStock(symbol, companyName) {

    try {

        const token = localStorage.getItem("token");

        const response = await fetch(`${API_URL}/api/watchlist`, {

            method: "POST",

            headers: {

                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`

            },

            body: JSON.stringify({

                symbol,
                companyName,
                exchange: "NSE"

            })

        });

        const data = await response.json();

        if (!response.ok) {

            alert(data.message || "Unable to add stock.");

            return;

        }

        document.getElementById("stock-symbol").value = "";
        document.getElementById("company-name").value = "";

        loadWatchlist();

    }

    catch (error) {

        console.error(error);

        alert("Unable to connect to server.");

    }

}

// ==========================================
// Load Watchlist
// ==========================================

async function loadWatchlist() {

    try {

        const token = localStorage.getItem("token");

        const response = await fetch(`${API_URL}/api/watchlist`, {

            headers: {

                Authorization: `Bearer ${token}`

            }

        });

        const stocks = await response.json();

        const container = document.getElementById("watchlist");

        container.innerHTML = "";

        if (!stocks.length) {

            container.innerHTML = `

                <div class="empty-watchlist">

                    No stocks added yet.

                </div>

            `;

            return;

        }

        stocks.forEach(stock => {

            container.innerHTML += `

                <div class="watch-card">

                    <div>

                        <h3>${stock.symbol}</h3>

                        <p>${stock.company_name}</p>

                    </div>

                    <button
                        class="delete-btn"
                        onclick="deleteStock(${stock.id})">

                        Remove

                    </button>

                </div>

            `;

        });

    }

    catch (error) {

        console.error(error);

    }

}

// ==========================================
// Delete Stock
// ==========================================

async function deleteStock(id) {

    try {

        const token = localStorage.getItem("token");

        const response = await fetch(`${API_URL}/api/watchlist/${id}`, {

            method: "DELETE",

            headers: {

                Authorization: `Bearer ${token}`

            }

        });

        const data = await response.json();

        if (!response.ok) {

            alert(data.message);

            return;

        }

        loadWatchlist();

    }

    catch (error) {

        console.error(error);

    }

}

// ==========================================
// Add Button
// ==========================================

document.addEventListener("DOMContentLoaded", () => {

    loadWatchlist();

    document
        .getElementById("add-stock-btn")
        .addEventListener("click", () => {

            const symbol = document
                .getElementById("stock-symbol")
                .value
                .trim()
                .toUpperCase();

            const companyName = document
                .getElementById("company-name")
                .value
                .trim();

            if (!symbol || !companyName) {

                alert("Please enter stock details.");

                return;

            }

            addStock(symbol, companyName);

        });

});