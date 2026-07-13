// =========================================
// AI STOCK MONITOR - Dashboard JS
// =========================================

document.addEventListener("DOMContentLoaded", () => {

    loadDashboard();

});

async function loadDashboard() {

    const token = localStorage.getItem("token");

    if (!token) {

        window.location.href = "login.html";
        return;

    }

    try {

        const response = await fetch("/api/dashboard", {

            method: "GET",

            headers: {

                "Authorization": `Bearer ${token}`,
                "Content-Type": "application/json"

            }

        });

        if (!response.ok) {

            localStorage.removeItem("token");
            localStorage.removeItem("user");

            window.location.href = "login.html";

            return;

        }

        const data = await response.json();

        renderDashboard(data);

    }

    catch (error) {

        console.error(error);

        showDashboardError();

    }

}

function renderDashboard(data) {

    document.getElementById("welcome-user").textContent =
        `Welcome, ${data.user.full_name}`;

    document.getElementById("user-email").textContent =
        data.user.email;

    document.getElementById("watchlist-count").textContent =
        data.watchlistCount;

    document.getElementById("market-status").textContent =
        data.marketStatus;

    document.getElementById("last-updated").textContent =
        data.lastUpdated;

}

function showDashboardError() {

    document.getElementById("welcome-user").textContent =
        "Unable to load dashboard.";

}