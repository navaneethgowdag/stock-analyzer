const newsService = require("../services/newsService");

exports.getNews = async (req, res) => {

    try {

        const news = await newsService.getNews(req.user.id);

        res.json(news);

    }

    catch (err) {

        console.error(err);

        res.status(500).json({
            message: "Unable to fetch news."
        });

    }

};

