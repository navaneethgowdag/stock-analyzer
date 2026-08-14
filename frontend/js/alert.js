// ==========================================
// Alert API
// ==========================================

const ALERT_API_URL = "https://stock-analyzer-backend-server.onrender.com";

let allAlerts = [];


// ==========================================
// Load Alerts
// ==========================================

async function loadAlerts() {

    const container =
        document.getElementById("alerts-container");

    try {

        const token =
            localStorage.getItem("token");

        if (!token) {

            container.innerHTML = `
                <div class="alerts-empty">
                    Please login to view alerts.
                </div>
            `;

            return;
        }


        const response = await fetch(
            `${ALERT_API_URL}/api/alerts/latest?limit=20`,
            {
                method: "GET",

                headers: {
                    "Authorization": `Bearer ${token}`
                }
            }
        );


        const data =
            await response.json();


        if (!response.ok) {

            console.error(
                "Alert API error:",
                data
            );

            container.innerHTML = `
                <div class="alerts-empty">
                    Unable to load alerts.
                </div>
            `;

            return;
        }


        allAlerts =
            data.alerts || [];


        renderAlertStocks();


    } catch (error) {

        console.error(
            "Failed to load alerts:",
            error
        );

        container.innerHTML = `
            <div class="alerts-empty">
                Unable to load alerts.
            </div>
        `;
    }
}


// ==========================================
// Group alerts by stock
// ==========================================

function groupAlertsByStock(alerts) {

    const grouped = {};

    alerts.forEach(alert => {

        const ticker =
            alert.ticker;

        if (!ticker) {
            return;
        }

        if (!grouped[ticker]) {

            grouped[ticker] = [];

        }

        grouped[ticker].push(alert);

    });

    return grouped;
}


// ==========================================
// Render stock buttons
// ==========================================

function renderAlertStocks() {

    const container =
        document.getElementById(
            "alerts-container"
        );

    if (!allAlerts.length) {

        container.innerHTML = `
            <div class="alerts-empty">
                No important alerts right now.
            </div>
        `;

        return;
    }


    const grouped =
        groupAlertsByStock(allAlerts);


    container.innerHTML = "";


    Object.entries(grouped).forEach(
        ([ticker, alerts]) => {

            const latest =
                alerts[0];


            const severity =
                getSeverityClass(
                    latest.severity
                );


            const item =
                document.createElement("div");


            item.className =
                "alert-stock-item";


            item.innerHTML = `

                <div class="alert-stock-left">

                    <div
                        class="alert-severity-dot
                        ${severity.dot}">
                    </div>

                    <div>

                        <div class="alert-stock-symbol">
                            ${escapeHtml(ticker)}
                        </div>

                        <div class="alert-stock-count">
                            ${alerts.length}
                            alert${alerts.length !== 1 ? "s" : ""}
                        </div>

                    </div>

                </div>

                <div class="alert-stock-arrow">
                    →
                </div>

            `;


            item.addEventListener(
                "click",
                () => {

                    openAlertModal(
                        ticker,
                        alerts
                    );

                }
            );


            container.appendChild(item);

        }
    );
}


// ==========================================
// Open Alert Modal
// ==========================================

function openAlertModal(
    ticker,
    alerts
) {

    const modal =
        document.getElementById(
            "alert-modal"
        );

    const title =
        document.getElementById(
            "alert-modal-title"
        );

    const subtitle =
        document.getElementById(
            "alert-modal-subtitle"
        );

    const body =
        document.getElementById(
            "alert-modal-body"
        );


    title.textContent =
        ticker;


    subtitle.textContent =
        `${alerts.length} alert${alerts.length !== 1 ? "s" : ""}`;


    body.innerHTML = "";


    alerts.forEach(alert => {

        const severity =
            getSeverityClass(
                alert.severity
            );


        const detail =
            document.createElement("div");


        detail.className =
            "alert-detail";


        detail.innerHTML = `

            <div class="alert-detail-top">

                <span class="alert-detail-type">

                    ${escapeHtml(
                        alert.alert_type || "ALERT"
                    )}

                </span>

                <span
                    class="alert-detail-severity
                    ${severity.badge}">

                    ${escapeHtml(
                        alert.severity || "INFO"
                    )}

                </span>

            </div>


            <h4 class="alert-detail-title">

                ${escapeHtml(
                    alert.title || "Stock Alert"
                )}

            </h4>


            <p class="alert-detail-message">

                ${escapeHtml(
                    alert.message || "No details available."
                )}

            </p>


            <div class="alert-detail-footer">

                <span>
                    ${formatAlertValue(alert.value)}
                </span>

                <span>
                    ${formatAlertDate(
                        alert.created_at
                    )}
                </span>

            </div>

        `;


        body.appendChild(detail);

    });


    modal.classList.add("active");

    document.body.style.overflow = "hidden";
}


// ==========================================
// Close Modal
// ==========================================

function closeAlertModal() {

    const modal =
        document.getElementById(
            "alert-modal"
        );

    modal.classList.remove("active");

    document.body.style.overflow = "";
}


// ==========================================
// Modal Events
// ==========================================

document.addEventListener(
    "DOMContentLoaded",
    () => {

        loadAlerts();


        const closeButton =
            document.getElementById(
                "close-alert-modal"
            );


        closeButton.addEventListener(
            "click",
            closeAlertModal
        );


        const modal =
            document.getElementById(
                "alert-modal"
            );


        modal.addEventListener(
            "click",
            event => {

                if (
                    event.target === modal
                ) {

                    closeAlertModal();

                }

            }
        );


        document.addEventListener(
            "keydown",
            event => {

                if (
                    event.key === "Escape"
                ) {

                    closeAlertModal();

                }

            }
        );

    }
);


// ==========================================
// Severity
// ==========================================

function getSeverityClass(severity) {

    const value =
        String(
            severity || ""
        ).toUpperCase();


    if (
        value.includes("POSITIVE") ||
        value.includes("HIGH_POSITIVE")
    ) {

        return {
            dot: "alert-severity-positive",
            badge: "alert-positive"
        };

    }


    if (
        value.includes("NEGATIVE") ||
        value.includes("HIGH_NEGATIVE")
    ) {

        return {
            dot: "alert-severity-negative",
            badge: "alert-negative"
        };

    }


    if (
        value.includes("WARNING") ||
        value.includes("HIGH")
    ) {

        return {
            dot: "alert-severity-warning",
            badge: "alert-warning"
        };

    }


    return {
        dot: "alert-severity-neutral",
        badge: "alert-neutral"
    };
}


// ==========================================
// Format value
// ==========================================

function formatAlertValue(value) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {

        return "";

    }

    const number =
        Number(value);


    if (!Number.isNaN(number)) {

        return `Value: ${number.toFixed(2)}`;

    }


    return escapeHtml(
        String(value)
    );
}


// ==========================================
// Format date
// ==========================================

function formatAlertDate(date) {

    if (!date) {
        return "";
    }


    const parsed =
        new Date(date);


    if (Number.isNaN(parsed.getTime())) {
        return "";
    }


    return parsed.toLocaleString(
        "en-IN",
        {
            dateStyle: "medium",
            timeStyle: "short"
        }
    );
}


// ==========================================
// HTML Escape
// ==========================================

function escapeHtml(value) {

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}