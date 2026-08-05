const NEWS_API = "http://localhost:5000/api/news";


async function loadNews() {

    try {

        const token = localStorage.getItem("token");

        const response = await fetch(NEWS_API, {

            headers: {

                Authorization: `Bearer ${token}`

            }

        });

        const news = await response.json();

        renderNews(news);

    }

    catch(err){

        console.error(err);

    }

}

function renderNews(news){

    const container = document.getElementById("news-container");

    container.innerHTML = "";

    if(news.length===0){

        container.innerHTML = `
            <p class="widget-message">
                No news available.
            </p>
        `;

        return;

    }

    // Group by ticker

    const grouped = {};

    news.forEach(item=>{

        if(!grouped[item.ticker]){

            grouped[item.ticker]=[];

        }

        grouped[item.ticker].push(item);

    });

    Object.keys(grouped).forEach(ticker=>{

        let html=`

        <div class="stock-news-group">

            <div class="stock-news-header">

                📈 ${ticker}

            </div>

        `;

        grouped[ticker].forEach(article=>{

            const badge =
                article.label==="positive"
                ? "positive"
                : article.label==="negative"
                ? "negative"
                : "neutral";

            html+=`

            <div class="news-item">

                <div class="news-top">

                    <span class="badge ${badge}">
                        ${article.label}
                    </span>

                    <span class="confidence">

                        ${(article.confidence*100).toFixed(1)}%

                    </span>

                </div>

                <div class="headline">

                    ${article.headline}

                </div>

                <div class="news-time">

                    ${new Date(article.created_at).toLocaleString()}

                </div>

            </div>

            `;

        });

        html+=`</div>`;

        container.innerHTML+=html;

    });

}

window.loadNews = loadNews;

document.addEventListener("DOMContentLoaded",loadNews);