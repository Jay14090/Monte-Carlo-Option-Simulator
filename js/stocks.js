// NIFTY 50 Stock Universe — NSE Tickers & Metadata
const NIFTY50 = [
  { symbol: "RELIANCE",   name: "Reliance Industries",          sector: "Energy",       typicalVol: 0.26 },
  { symbol: "TCS",        name: "Tata Consultancy Services",    sector: "IT",           typicalVol: 0.22 },
  { symbol: "HDFCBANK",   name: "HDFC Bank",                    sector: "Banking",      typicalVol: 0.24 },
  { symbol: "ICICIBANK",  name: "ICICI Bank",                   sector: "Banking",      typicalVol: 0.28 },
  { symbol: "INFY",       name: "Infosys",                      sector: "IT",           typicalVol: 0.25 },
  { symbol: "BHARTIARTL", name: "Bharti Airtel",                sector: "Telecom",      typicalVol: 0.30 },
  { symbol: "ITC",        name: "ITC Limited",                  sector: "FMCG",         typicalVol: 0.22 },
  { symbol: "KOTAKBANK",  name: "Kotak Mahindra Bank",          sector: "Banking",      typicalVol: 0.25 },
  { symbol: "LT",         name: "Larsen & Toubro",              sector: "Engineering",  typicalVol: 0.27 },
  { symbol: "HINDUNILVR", name: "Hindustan Unilever",           sector: "FMCG",         typicalVol: 0.20 },
  { symbol: "AXISBANK",   name: "Axis Bank",                    sector: "Banking",      typicalVol: 0.30 },
  { symbol: "SBIN",       name: "State Bank of India",          sector: "Banking",      typicalVol: 0.32 },
  { symbol: "BAJFINANCE", name: "Bajaj Finance",                sector: "NBFC",         typicalVol: 0.35 },
  { symbol: "MARUTI",     name: "Maruti Suzuki India",          sector: "Auto",         typicalVol: 0.26 },
  { symbol: "HCLTECH",    name: "HCL Technologies",             sector: "IT",           typicalVol: 0.24 },
  { symbol: "SUNPHARMA",  name: "Sun Pharmaceutical",           sector: "Pharma",       typicalVol: 0.28 },
  { symbol: "ADANIPORTS", name: "Adani Ports & SEZ",            sector: "Infrastructure",typicalVol:0.38 },
  { symbol: "TATAMOTORS", name: "Tata Motors",                  sector: "Auto",         typicalVol: 0.40 },
  { symbol: "TITAN",      name: "Titan Company",                sector: "Consumer",     typicalVol: 0.29 },
  { symbol: "WIPRO",      name: "Wipro",                        sector: "IT",           typicalVol: 0.26 },
  { symbol: "ULTRACEMCO", name: "UltraTech Cement",             sector: "Cement",       typicalVol: 0.25 },
  { symbol: "NTPC",       name: "NTPC Limited",                 sector: "Power",        typicalVol: 0.28 },
  { symbol: "POWERGRID",  name: "Power Grid Corporation",       sector: "Power",        typicalVol: 0.25 },
  { symbol: "TATASTEEL",  name: "Tata Steel",                   sector: "Metals",       typicalVol: 0.38 },
  { symbol: "JSWSTEEL",   name: "JSW Steel",                    sector: "Metals",       typicalVol: 0.36 },
  { symbol: "HINDALCO",   name: "Hindalco Industries",          sector: "Metals",       typicalVol: 0.34 },
  { symbol: "ONGC",       name: "Oil & Natural Gas Corporation",sector: "Energy",       typicalVol: 0.30 },
  { symbol: "DRREDDY",    name: "Dr. Reddy's Laboratories",     sector: "Pharma",       typicalVol: 0.28 },
  { symbol: "CIPLA",      name: "Cipla",                        sector: "Pharma",       typicalVol: 0.27 },
  { symbol: "GRASIM",     name: "Grasim Industries",            sector: "Diversified",  typicalVol: 0.26 },
  { symbol: "NESTLEIND",  name: "Nestlé India",                 sector: "FMCG",         typicalVol: 0.19 },
  { symbol: "BRITANNIA",  name: "Britannia Industries",         sector: "FMCG",         typicalVol: 0.22 },
  { symbol: "DIVISLAB",   name: "Divi's Laboratories",          sector: "Pharma",       typicalVol: 0.30 },
  { symbol: "APOLLOHOSP", name: "Apollo Hospitals Enterprise",  sector: "Healthcare",   typicalVol: 0.32 },
  { symbol: "BAJAJ-AUTO", name: "Bajaj Auto",                   sector: "Auto",         typicalVol: 0.23 },
  { symbol: "BAJAJFINSV", name: "Bajaj Finserv",                sector: "NBFC",         typicalVol: 0.32 },
  { symbol: "EICHERMOT",  name: "Eicher Motors",                sector: "Auto",         typicalVol: 0.27 },
  { symbol: "HEROMOTOCO", name: "Hero MotoCorp",                sector: "Auto",         typicalVol: 0.24 },
  { symbol: "HDFCLIFE",   name: "HDFC Life Insurance",          sector: "Insurance",    typicalVol: 0.26 },
  { symbol: "SBILIFE",    name: "SBI Life Insurance",           sector: "Insurance",    typicalVol: 0.27 },
  { symbol: "SHRIRAMFIN", name: "Shriram Finance",              sector: "NBFC",         typicalVol: 0.34 },
  { symbol: "INDUSINDBK", name: "IndusInd Bank",                sector: "Banking",      typicalVol: 0.33 },
  { symbol: "ASIANPAINT", name: "Asian Paints",                 sector: "Consumer",     typicalVol: 0.22 },
  { symbol: "BPCL",       name: "Bharat Petroleum Corporation", sector: "Energy",       typicalVol: 0.33 },
  { symbol: "COALINDIA",  name: "Coal India",                   sector: "Mining",       typicalVol: 0.28 },
  { symbol: "ADANIENT",   name: "Adani Enterprises",            sector: "Conglomerate", typicalVol: 0.45 },
  { symbol: "LTIM",       name: "LTIMindtree",                  sector: "IT",           typicalVol: 0.29 },
  { symbol: "TATACONSUM", name: "Tata Consumer Products",       sector: "FMCG",         typicalVol: 0.27 },
  { symbol: "TECHM",      name: "Tech Mahindra",                sector: "IT",           typicalVol: 0.30 },
  { symbol: "UPL",        name: "UPL Limited",                  sector: "Agrochemicals",typicalVol: 0.35 },
];

// Fallback prices for offline/demo mode (approximate values in ₹, Feb 2026)
const FALLBACK_PRICES = {
  RELIANCE:1285,TCS:3780,HDFCBANK:1640,ICICIBANK:1220,INFY:1870,
  BHARTIARTL:1710,ITC:415,KOTAKBANK:1870,LT:3450,HINDUNILVR:2320,
  AXISBANK:1050,SBIN:770,BAJFINANCE:6950,MARUTI:11200,HCLTECH:1720,
  SUNPHARMA:1790,ADANIPORTS:1165,TATAMOTORS:690,TITAN:3320,WIPRO:310,
  ULTRACEMCO:11400,NTPC:335,POWERGRID:295,TATASTEEL:150,JSWSTEEL:965,
  HINDALCO:640,ONGC:260,DRREDDY:1195,CIPLA:1490,GRASIM:2530,
  NESTLEIND:2250,BRITANNIA:5180,DIVISLAB:5250,APOLLOHOSP:6740,
  "BAJAJ-AUTO":8750,BAJAJFINSV:1680,EICHERMOT:5180,HEROMOTOCO:4180,
  HDFCLIFE:625,SBILIFE:1565,SHRIRAMFIN:580,INDUSINDBK:990,
  ASIANPAINT:2290,BPCL:285,COALINDIA:390,ADANIENT:2435,LTIM:4960,
  TATACONSUM:918,TECHM:1580,UPL:520,
};

function getStockBySymbol(symbol) {
  return NIFTY50.find(s => s.symbol === symbol) || null;
}

function getFallbackPrice(symbol) {
  return FALLBACK_PRICES[symbol] || 1000;
}
