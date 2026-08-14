# Stock Analyzer

An AI-powered full-stack stock analysis platform that combines market
data, machine-learning predictions, financial-news sentiment,
watchlists, portfolio analytics, alerts, and user feedback.

## Overview

Stock Analyzer was developed as three cooperating layers:

-   **Frontend:** HTML, CSS and JavaScript dashboard.
-   **Node.js backend:** Express REST API for authentication,
    watchlists, portfolio data, market data, alerts, feedback and
    suggestions.
-   **Python AI backend:** scheduled stock-analysis pipeline that reads
    the watchlist, fetches market/news data, runs ML and FinBERT
    analysis, and updates PostgreSQL.

The production database uses PostgreSQL through Neon.

------------------------------------------------------------------------

## Main Features

### Authentication

The application includes:

-   Registration
-   Login
-   JWT authentication
-   Protected API routes
-   Authentication middleware
-   Token/session expiration handling
-   API request rate limiting

Protected requests use:

``` text
Authorization: Bearer <JWT_TOKEN>
```

### Watchlist

Users can maintain a personal stock watchlist containing:

-   Stock symbol
-   Company name
-   Exchange
-   User ownership

The backend follows:

``` text
Route → Controller → Service → Database
```

### AI Stock Predictions

The Python pipeline processes stocks from the database watchlist.

``` text
Watchlist
   ↓
Yahoo Finance
   ↓
Historical OHLCV
   ↓
Feature Engineering
   ↓
Technical Indicators
   ↓
ML Model
   ↓
Prediction Probability
   ↓
News + FinBERT Sentiment
   ↓
Hybrid Score
   ↓
BUY / HOLD / SELL
   ↓
PostgreSQL
```

Prediction data includes:

-   Current price
-   Previous close
-   Probability of upward movement
-   Prediction direction
-   Sentiment label
-   Sentiment score
-   Combined/hybrid score
-   Recommendation
-   Last updated time

### Previous Close

`previous_close` was added to the predictions data so the Market section
can compare the current price with the previous close.

The Python pipeline first derives a previous close from historical data
and attempts to use Yahoo Finance's live previous-close value when
available.

### Market Section

The Market section displays watchlist stocks with:

-   Current price
-   Previous close
-   Up/down movement
-   Stock symbol

### Financial News

The News section provides stock-specific financial news and a stock
selection/popup interface.

The Python pipeline fetches financial headlines for processed stocks.

### FinBERT Sentiment

Financial headlines are analyzed with FinBERT.

``` text
Financial Headlines
        ↓
      FinBERT
        ↓
Sentiment Score + Label
        ↓
Hybrid Prediction Score
```

The sentiment contributes to the final recommendation.

### Alerts

The Alerts section displays important stock events.

Alerts include:

-   News alerts
-   Price alerts
-   Stock/ticker
-   Alert title
-   Message
-   Severity
-   Value
-   Reference information

Price alerts are generated when the price movement crosses the
configured threshold of approximately **±3%**.

Examples:

``` text
PRICE_SURGE
PRICE_DROP
```

### Price History

The Python pipeline stores current prices in a price-history table.

The flow is:

``` text
Current Price
    ↓
Store Price History
    ↓
Read Previous Recorded Price
    ↓
Calculate Percentage Change
    ↓
Generate Alert if Threshold Crossed
```

### Portfolio

The Portfolio section summarizes watchlist/prediction information,
including:

-   Total stocks
-   BUY count
-   HOLD count
-   SELL count
-   Average confidence
-   Average price
-   Overall sentiment
-   Last updated time

The portfolio backend uses:

``` text
portfolioRoutes
      ↓
portfolioController
      ↓
portfolioService
      ↓
PostgreSQL
```

### Feedback and Suggestions

Users can:

-   Give feedback
-   Suggest an idea
-   Select a category
-   Enter a message
-   Submit the form

The UI includes modal forms, character counting, success/error messages
and responsive styling.

The feedback UI supports both dark and light themes.

### Dark and Light Theme

Theme-aware CSS variables are used for components such as:

``` css
--text-primary
--text-secondary
--card-bg
--border-color
--surface-secondary
--surface-hover
--input-bg
--input-focus-bg
--accent-color
--accent-soft
--modal-overlay
--shadow-color
```

### Automatic Refresh

Because the Python job updates the database independently of the
browser, the dashboard needs to request fresh data.

A full-page refresh can be scheduled with:

``` javascript
setInterval(() => {
    location.reload();
}, 30000);
```

`30000` milliseconds equals **30 seconds**.

------------------------------------------------------------------------

# Architecture

``` text
                         ┌──────────────────┐
                         │     Frontend     │
                         │    HTML/CSS/JS   │
                         └────────┬─────────┘
                                  │
                              REST API
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Node.js/Express │
                         │      Backend     │
                         └────────┬─────────┘
                                  │
                              PostgreSQL
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   Neon Database  │
                         └────────▲─────────┘
                                  │
                            Database updates
                                  │
                         ┌────────┴─────────┐
                         │  GitHub Actions  │
                         │   Python Job     │
                         └────────┬─────────┘
                                  │
                     ┌────────────┼────────────┐
                     ▼            ▼            ▼
                Yahoo Finance  ML Models    FinBERT
```

The frontend does not run the AI pipeline itself.

The Node.js backend serves application APIs.

The Python job performs the heavier analysis and writes results into
PostgreSQL.

------------------------------------------------------------------------

# Backend Architecture

The Node.js backend uses a modular structure:

``` text
backend/
└── src/
    ├── controllers/
    ├── middleware/
    ├── models/
    ├── routes/
    ├── services/
    ├── app.js
    └── server.js
```

Responsibilities:

### Routes

Define API endpoints.

### Controllers

Handle HTTP requests and responses.

### Services

Contain application/business logic.

### Models

Execute database queries.

### Middleware

Handle authentication and request protection.

------------------------------------------------------------------------

# Authentication Flow

``` text
Login/Register
      ↓
Node.js API
      ↓
PostgreSQL
      ↓
JWT
      ↓
Frontend stores token
      ↓
Protected API request
      ↓
JWT Middleware
      ↓
req.user
      ↓
Controller
```

The authentication middleware reads the `Authorization` header, extracts
the Bearer token, verifies it using `JWT_SECRET`, and places the decoded
user data in `req.user`.

------------------------------------------------------------------------

# Python AI Pipeline

The Python backend uses libraries including:

-   NumPy
-   Pandas
-   yfinance
-   TA
-   scikit-learn
-   XGBoost
-   Joblib
-   psycopg2
-   python-dotenv
-   APScheduler
-   Hugging Face / Transformers
-   FinBERT

The model artifacts are stored under the AI backend's model directory.

A typical structure is:

``` text
ai-backend/
├── models/
├── pred.py
└── requirements.txt
```

The processing flow is:

``` text
Read Watchlist
      ↓
Fetch Historical Data
      ↓
Cache/Update Stock History
      ↓
Create Features
      ↓
Load ML Model
      ↓
Predict Direction
      ↓
Calculate Probability
      ↓
Fetch News
      ↓
FinBERT Sentiment
      ↓
Hybrid Score
      ↓
Recommendation
      ↓
Update Predictions
      ↓
Update News Sentiment
      ↓
Update Price History
      ↓
Generate Alerts
```

------------------------------------------------------------------------

# Database

PostgreSQL is used as the central data store.

The application stores information for:

-   Users
-   Watchlists
-   Predictions
-   Current prices
-   Previous closes
-   News sentiment
-   Price history
-   Alerts
-   Feedback
-   Suggestions
-   Portfolio-related data

The Node.js backend uses:

``` text
pg
```

The Python backend uses:

``` text
psycopg2
```

------------------------------------------------------------------------

# Deployment

The project was deployed as separate workloads.

## Frontend

The frontend is deployed as a static web application.

## Node.js Backend

The Express backend is deployed as a web service.

## Python AI Job

The Python AI pipeline was moved away from the always-running web
service because the heavier ML/FinBERT workload exceeded the available
memory on the previous hosting setup.

The Python process does not need to expose an HTTP API. Its job is
simply:

``` text
Start
 ↓
Process stocks
 ↓
Update database
 ↓
Finish
```

Therefore GitHub Actions is used to execute the Python workload.

------------------------------------------------------------------------

# GitHub Actions

The GitHub Actions workflow:

1.  Checks out the repository.
2.  Sets up Python.
3.  Installs `requirements.txt`.
4.  Provides database/environment secrets.
5.  Runs:

``` bash
python -u pred.py
```

The overall production flow becomes:

``` text
GitHub Actions
      ↓
   pred.py
      ↓
Yahoo Finance + ML + FinBERT
      ↓
PostgreSQL / Neon
      ↓
Node.js REST API
      ↓
Frontend Dashboard
```

------------------------------------------------------------------------

# Environment Variables

Sensitive configuration must not be committed.

Typical backend variables include:

``` text
DATABASE_URL
JWT_SECRET
PORT
```

The AI backend requires the database connection and any other
model/service configuration used by the pipeline.

Use hosting-platform secrets and GitHub Actions secrets for production
values.

------------------------------------------------------------------------

# .gitignore

Recommended entries include:

``` gitignore
.env
.env.*
!.env.example

node_modules/

.venv/
venv/
__pycache__/
*.py[cod]

*.log

.DS_Store
Thumbs.db

.vscode/
.idea/

dist/
build/

*.sqlite
*.db
```

Large model artifacts should also be managed intentionally instead of
committing unnecessary generated files.

------------------------------------------------------------------------

# Local Setup

## 1. Clone

``` bash
git clone <repository-url>
cd stock-analyzer
```

## 2. Node.js Backend

``` bash
cd backend
npm install
node src/server.js
```

Development backend:

``` text
http://localhost:5000
```

## 3. Python Environment

From the AI backend:

``` bash
python -m venv .venv
```

Windows PowerShell:

``` powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Run the AI job:

``` bash
python pred.py
```

## 4. Frontend

Configure the frontend API base URL to point to the local Node.js server
and open the application through the configured static/development
server.

------------------------------------------------------------------------

# Development History

The application was built incrementally.

``` text
Project setup
     ↓
Authentication
     ↓
PostgreSQL connection
     ↓
Watchlist
     ↓
Prediction pipeline
     ↓
Prediction database
     ↓
News
     ↓
FinBERT sentiment
     ↓
Market section
     ↓
Previous close
     ↓
Price history
     ↓
Alerts
     ↓
Portfolio
     ↓
Feedback & suggestions
     ↓
Dark/light theme
     ↓
Automatic refresh
     ↓
Frontend deployment
     ↓
Backend deployment
     ↓
AI workload deployment
     ↓
GitHub Actions automation
```

------------------------------------------------------------------------

# Important Debugging Issues Solved

## Express route handler error

An error such as:

``` text
TypeError: argument handler must be a function
```

was caused by a route receiving an undefined/non-function handler.
Controller exports and route imports must match.

## PostgreSQL connection refused

``` text
ECONNREFUSED 127.0.0.1:5432
```

indicates that the application was trying to connect to a local
PostgreSQL server that was unavailable.

## Database DNS failure

``` text
ENOTFOUND
```

indicates that the configured database hostname could not be resolved.

## Frontend JSON error

``` text
Unexpected token '<', "<!DOCTYPE "... is not valid JSON
```

occurs when the frontend expects JSON but receives an HTML error page,
commonly because the API URL is wrong or the backend returned a 404.

## Incorrect API URL

Production URLs must not accidentally contain:

``` text
//api/market
```

when the intended path is:

``` text
/api/market
```

## Undefined token

A frontend error such as:

``` text
ReferenceError: token is not defined
```

means the authentication token was not retrieved before constructing the
Authorization header.

## Python price-alert error

The Python job encountered:

``` text
generate_price_alert() missing 1 required positional argument: 'current_price'
```

The function expects:

``` python
generate_price_alert(
    conn,
    ticker,
    current_price
)
```

The corrected processing order is:

``` python
insert_price_history(
    conn,
    ticker,
    current_price
)

generate_price_alert(
    conn,
    ticker,
    current_price
)
```

------------------------------------------------------------------------

# Security

Production deployments should:

-   Keep `.env` files out of Git.
-   Keep JWT secrets private.
-   Keep database credentials private.
-   Use HTTPS.
-   Protect private endpoints with JWT.
-   Apply rate limiting.
-   Validate user input.
-   Use parameterized SQL queries.
-   Avoid returning raw database errors to clients.
-   Store deployment secrets in platform/GitHub secret storage.

------------------------------------------------------------------------

# Future Improvements

Possible next steps:

-   WebSocket/SSE live updates instead of full-page refreshes
-   Redis-based distributed rate limiting
-   Better background-job monitoring
-   Retry handling for external APIs
-   Pinned ML dependency versions
-   Model version management
-   Prediction accuracy tracking
-   Historical prediction performance
-   More technical indicators
-   More financial data providers
-   User-specific alert preferences
-   Email/push notifications
-   Alert deduplication
-   Backtesting
-   Automated model retraining
-   Automated backend/API tests
-   Python pipeline tests
-   Proper database migrations

------------------------------------------------------------------------

# Technology Stack

## Frontend

-   HTML5
-   CSS3
-   JavaScript
-   Fetch API
-   Responsive UI
-   Dark/light theme

## Backend

-   Node.js
-   Express
-   JWT
-   PostgreSQL
-   pg
-   REST APIs
-   Middleware
-   Rate limiting

## AI/Data

-   Python
-   NumPy
-   Pandas
-   yfinance
-   TA
-   scikit-learn
-   XGBoost
-   Joblib
-   Transformers / Hugging Face
-   FinBERT
-   psycopg2
-   APScheduler
-   python-dotenv

## Database

-   PostgreSQL
-   Neon

## Automation

-   GitHub
-   GitHub Actions

------------------------------------------------------------------------

# Repository Structure

A typical repository structure is:

``` text
stock-analyzer/
│
├── frontend/
│   ├── index.html
│   ├── css/
│   └── js/
│
├── backend/
│   ├── src/
│   │   ├── controllers/
│   │   ├── middleware/
│   │   ├── models/
│   │   ├── routes/
│   │   ├── services/
│   │   ├── app.js
│   │   └── server.js
│   │
│   ├── package.json
│   └── .env
│
├── ai-backend/
│   ├── models/
│   ├── pred.py
│   └── requirements.txt
│
├── .github/
│   └── workflows/
│
├── .gitignore
└── README.md
```

------------------------------------------------------------------------

# End-to-End Summary

``` text
                         USER
                           │
                           ▼
                  ┌─────────────────┐
                  │    FRONTEND     │
                  │    HTML/CSS/JS  │
                  └────────┬────────┘
                           │
                      JWT + REST
                           │
                           ▼
                  ┌─────────────────┐
                  │ NODE.JS/EXPRESS │
                  │     BACKEND     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ POSTGRESQL/NEON │
                  └────────▲────────┘
                           │
                     Database Updates
                           │
                  ┌────────┴────────┐
                  │ GITHUB ACTIONS  │
                  │   PYTHON JOB    │
                  └────────┬────────┘
                           │
                ┌──────────┼──────────┐
                ▼          ▼          ▼
             Yahoo       ML Models   FinBERT
             Finance                  News
```

The final system separates presentation, API/business logic, persistent
storage, and computationally intensive AI processing into independent
components.

------------------------------------------------------------------------

# Author

**Navaneeth Gowda**

Stock Analyzer --- full-stack AI-powered stock analysis project.
