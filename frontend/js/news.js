// ==========================================
// STOCK NEWS
// ==========================================

const NEWS_API = "https://stock-analyzer-backend-server.onrender.com/api/news";

let stockNews = {};


// ==========================================
// Load News
// ==========================================

async function loadNews() {

    try {

        const token = localStorage.getItem("token");

        if (!token) {

            console.warn("No authentication token found.");

            return;

        }

        const response = await fetch(NEWS_API, {

            method: "GET",

            headers: {

                "Authorization": `Bearer ${token}`

            }

        });

        if (!response.ok) {

            console.error(
                "News API error:",
                response.status
            );

            return;

        }

        const news = await response.json();

        groupNewsByStock(news);

        renderStockButtons();

    }

    catch (error) {

        console.error(
            "Unable to load news:",
            error
        );

    }

}


// ==========================================
// Group News By Stock
// ==========================================

function groupNewsByStock(news) {

    stockNews = {};

    news.forEach(article => {

        const ticker = article.ticker;

        if (!ticker) {
            return;
        }

        if (!stockNews[ticker]) {

            stockNews[ticker] = [];

        }

        stockNews[ticker].push(article);

    });

}


// ==========================================
// Render Stock Buttons
// ==========================================

function renderStockButtons() {

    const container =
        document.getElementById("news-stock-list");

    if (!container) {
        return;
    }

    container.innerHTML = "";

    const stocks = Object.keys(stockNews);

    if (stocks.length === 0) {

        container.innerHTML = `
            <div class="news-empty">
                No stock news available.
            </div>
        `;

        return;

    }

    stocks.forEach(ticker => {

        const count = stockNews[ticker].length;

        const button =
            document.createElement("button");

        button.type = "button";

        button.className = "news-stock-button";

        button.innerHTML = `

            <div class="news-stock-left">

                <div class="news-stock-icon">

                    <i class="bi bi-newspaper"></i>

                </div>

                <div>

                    <div class="news-stock-name">
                        ${escapeHTML(ticker)}
                    </div>

                    <div class="news-stock-count">
                        ${count}
                        ${count === 1 ? "headline" : "headlines"}
                    </div>

                </div>

            </div>

            <i class="bi bi-chevron-right news-arrow"></i>

        `;

        button.addEventListener(
            "click",
            () => openNewsModal(ticker)
        );

        container.appendChild(button);

    });

}


// ==========================================
// Open News Modal
// ==========================================

function openNewsModal(ticker) {

    const modal =
        document.getElementById("news-modal");

    const tickerElement =
        document.getElementById("modal-ticker");

    const newsList =
        document.getElementById("modal-news-list");

    if (!modal || !tickerElement || !newsList) {
        return;
    }

    tickerElement.textContent = ticker;

    newsList.innerHTML = "";

    const articles = stockNews[ticker] || [];

    if (articles.length === 0) {

        newsList.innerHTML = `
            <div class="news-empty">
                No news available for ${escapeHTML(ticker)}.
            </div>
        `;

    }
    else {

        articles.forEach(article => {

            const sentiment =
                getSentimentClass(article.label);

            const confidence =
                Number(article.confidence);

            const confidenceText =
                Number.isFinite(confidence)
                    ? `${(confidence * 100).toFixed(1)}%`
                    : "N/A";

            const date =
                article.created_at
                    ? new Date(article.created_at)
                        .toLocaleString()
                    : "";

            newsList.innerHTML += `

                <div class="news-article">

                    <div class="news-article-headline">

                        ${escapeHTML(
                            article.headline || "No headline"
                        )}

                    </div>

                    <div class="news-article-meta">

                        <span class="
                            news-sentiment
                            ${sentiment}
                        ">

                            ${escapeHTML(
                                article.label || "Neutral"
                            )}

                        </span>

                        <span>
                            Confidence: ${confidenceText}
                        </span>

                        <span>
                            ${escapeHTML(date)}
                        </span>

                    </div>

                </div>

            `;

        });

    }

    modal.classList.add("active");

    document.body.style.overflow = "hidden";

}


// ==========================================
// Close Modal
// ==========================================

function closeNewsModal() {

    const modal =
        document.getElementById("news-modal");

    if (!modal) {
        return;
    }

    modal.classList.remove("active");

    document.body.style.overflow = "";

}


// ==========================================
// Sentiment Class
// ==========================================

function getSentimentClass(label) {

    if (!label) {
        return "neutral";
    }

    const value =
        label.toLowerCase();

    if (
        value.includes("positive") ||
        value.includes("bullish")
    ) {

        return "positive";

    }

    if (
        value.includes("negative") ||
        value.includes("bearish")
    ) {

        return "negative";

    }

    return "neutral";

}


// ==========================================
// Escape HTML
// ==========================================

function escapeHTML(value) {

    const div =
        document.createElement("div");

    div.textContent =
        String(value ?? "");

    return div.innerHTML;

}


// ==========================================
// Events
// ==========================================

document.addEventListener(
    "DOMContentLoaded",
    () => {

        loadNews();

        const closeButton =
            document.getElementById("news-close");

        if (closeButton) {

            closeButton.addEventListener(
                "click",
                closeNewsModal
            );

        }

        const modal =
            document.getElementById("news-modal");

        if (modal) {

            modal.addEventListener(
                "click",
                event => {

                    if (event.target === modal) {

                        closeNewsModal();

                    }

                }
            );

        }

        document.addEventListener(
            "keydown",
            event => {

                if (event.key === "Escape") {

                    closeNewsModal();

                }

            }
        );

    }
);


// ==========================================
// Make Available To Watchlist
// ==========================================

window.loadNews = loadNews;