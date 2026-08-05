const nifty50 = [
    { symbol: "ADANIENT.NS", company: "Adani Enterprises Ltd." },
    { symbol: "ADANIPORTS.NS", company: "Adani Ports and SEZ Ltd." },
    { symbol: "APOLLOHOSP.NS", company: "Apollo Hospitals Enterprise Ltd." },
    { symbol: "ASIANPAINT.NS", company: "Asian Paints Ltd." },
    { symbol: "AXISBANK.NS", company: "Axis Bank Ltd." },
    { symbol: "BAJAJ-AUTO.NS", company: "Bajaj Auto Ltd." },
    { symbol: "BAJFINANCE.NS", company: "Bajaj Finance Ltd." },
    { symbol: "BAJAJFINSV.NS", company: "Bajaj Finserv Ltd." },
    { symbol: "BEL.NS", company: "Bharat Electronics Ltd." },
    { symbol: "BHARTIARTL.NS", company: "Bharti Airtel Ltd." },
    { symbol: "CIPLA.NS", company: "Cipla Ltd." },
    { symbol: "COALINDIA.NS", company: "Coal India Ltd." },
    { symbol: "DRREDDY.NS", company: "Dr. Reddy's Laboratories Ltd." },
    { symbol: "EICHERMOT.NS", company: "Eicher Motors Ltd." },
    { symbol: "ETERNAL.NS", company: "Eternal Ltd." },
    { symbol: "GRASIM.NS", company: "Grasim Industries Ltd." },
    { symbol: "HCLTECH.NS", company: "HCL Technologies Ltd." },
    { symbol: "HDFCBANK.NS", company: "HDFC Bank Ltd." },
    { symbol: "HDFCLIFE.NS", company: "HDFC Life Insurance Co. Ltd." },
    { symbol: "HEROMOTOCO.NS", company: "Hero MotoCorp Ltd." },
    { symbol: "HINDALCO.NS", company: "Hindalco Industries Ltd." },
    { symbol: "HINDUNILVR.NS", company: "Hindustan Unilever Ltd." },
    { symbol: "ICICIBANK.NS", company: "ICICI Bank Ltd." },
    { symbol: "INDUSINDBK.NS", company: "IndusInd Bank Ltd." },
    { symbol: "INFY.NS", company: "Infosys Ltd." },
    { symbol: "ITC.NS", company: "ITC Ltd." },
    { symbol: "JIOFIN.NS", company: "Jio Financial Services Ltd." },
    { symbol: "JSWSTEEL.NS", company: "JSW Steel Ltd." },
    { symbol: "KOTAKBANK.NS", company: "Kotak Mahindra Bank Ltd." },
    { symbol: "LT.NS", company: "Larsen & Toubro Ltd." },
    { symbol: "M&M.NS", company: "Mahindra & Mahindra Ltd." },
    { symbol: "MARUTI.NS", company: "Maruti Suzuki India Ltd." },
    { symbol: "NESTLEIND.NS", company: "Nestlé India Ltd." },
    { symbol: "NTPC.NS", company: "NTPC Ltd." },
    { symbol: "ONGC.NS", company: "Oil & Natural Gas Corporation Ltd." },
    { symbol: "POWERGRID.NS", company: "Power Grid Corporation of India Ltd." },
    { symbol: "RELIANCE.NS", company: "Reliance Industries Ltd." },
    { symbol: "SBILIFE.NS", company: "SBI Life Insurance Co. Ltd." },
    { symbol: "SBIN.NS", company: "State Bank of India" },
    { symbol: "SHRIRAMFIN.NS", company: "Shriram Finance Ltd." },
    { symbol: "SUNPHARMA.NS", company: "Sun Pharmaceutical Industries Ltd." },
    { symbol: "TATACONSUM.NS", company: "Tata Consumer Products Ltd." },
    { symbol: "TATAMOTORS.NS", company: "Tata Motors Ltd." },
    { symbol: "TATASTEEL.NS", company: "Tata Steel Ltd." },
    { symbol: "TCS.NS", company: "Tata Consultancy Services Ltd." },
    { symbol: "TECHM.NS", company: "Tech Mahindra Ltd." },
    { symbol: "TITAN.NS", company: "Titan Company Ltd." },
    { symbol: "TRENT.NS", company: "Trent Ltd." },
    { symbol: "ULTRACEMCO.NS", company: "UltraTech Cement Ltd." },
    { symbol: "WIPRO.NS", company: "Wipro Ltd." }
];

const stockSelect = document.getElementById("stock-symbol");
const companyInput = document.getElementById("company-name");

nifty50.forEach(stock => {
    const option = document.createElement("option");
    option.value = stock.symbol;
    option.textContent = `${stock.symbol} - ${stock.company}`;
    stockSelect.appendChild(option);
});

stockSelect.addEventListener("change", () => {
    const selected = nifty50.find(s => s.symbol === stockSelect.value);

    companyInput.value = selected ? selected.company : "";
});