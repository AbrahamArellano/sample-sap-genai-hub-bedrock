
from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import requests
import random

app = FastAPI(title="Financial Data MCP Server")

class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    method: str
    params: Optional[Dict[str, Any]] = None

class JsonRpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    result: Optional[Any] = None

# Mock financial data including AWS and SAP
MOCK_STOCKS = {
    "AAPL": {"price": 195.50, "change": 2.1, "market_cap": "3.0T", "pe_ratio": 29.5, "sector": "Technology"},
    "MSFT": {"price": 420.25, "change": -0.8, "market_cap": "3.1T", "pe_ratio": 35.2, "sector": "Technology"},
    "AMZN": {"price": 155.75, "change": 1.5, "market_cap": "1.6T", "pe_ratio": 45.8, "sector": "Consumer Discretionary"},
    "SAP": {"price": 142.30, "change": 0.9, "market_cap": "175B", "pe_ratio": 22.4, "sector": "Software"},
    "TSLA": {"price": 248.42, "change": -3.2, "market_cap": "750B", "pe_ratio": 75.1, "sector": "Automotive"},
    "GOOGL": {"price": 175.80, "change": 1.2, "market_cap": "2.2T", "pe_ratio": 28.9, "sector": "Technology"},
}

ALPHA_VANTAGE_KEY = "demo"  # Replace with actual key if needed

def get_real_stock_data(symbol: str) -> Optional[Dict]:
    """Try Alpha Vantage, fallback to mock data"""
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
        response = requests.get(url, timeout=3)
        data = response.json()
        
        if "Global Quote" in data:
            quote = data["Global Quote"]
            return {
                "price": float(quote["05. price"]),
                "change": float(quote["09. change"]),
                "source": "live"
            }
    except:
        pass
    return None

TOOLS = [
    {
        "name": "get_stock_quote",
        "description": "Get current stock price and change for a symbol",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock symbol (e.g., AAPL, SAP, AMZN)"}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_company_overview", 
        "description": "Get company overview including market cap, P/E ratio, and sector",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock symbol"}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "calculate_financial_health",
        "description": "Calculate a simple financial health score (1-100)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock symbol"}
            },
            "required": ["symbol"]
        }
    }
]

def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Execute tools with financial logic"""
    symbol = arguments.get("symbol", "").upper()
    
    if name == "get_stock_quote":
        # Try live data first, fallback to mock
        real_data = get_real_stock_data(symbol)
        if real_data:
            return f"${real_data['price']:.2f} ({real_data['change']:+.2f}) [Live Data]"
        
        if symbol in MOCK_STOCKS:
            stock = MOCK_STOCKS[symbol]
            return f"${stock['price']:.2f} ({stock['change']:+.2f}%) [Mock Data]"
        return f"Stock symbol '{symbol}' not found. Try: {', '.join(list(MOCK_STOCKS.keys())[:5])}"
    
    elif name == "get_company_overview":
        if symbol in MOCK_STOCKS:
            stock = MOCK_STOCKS[symbol]
            return f"""Company: {symbol}
Market Cap: {stock['market_cap']}
P/E Ratio: {stock['pe_ratio']}
Sector: {stock['sector']}
Current Price: ${stock['price']:.2f}"""
        return f"Company '{symbol}' not found in database"
    
    elif name == "calculate_financial_health":
        if symbol in MOCK_STOCKS:
            stock = MOCK_STOCKS[symbol]
            # Simple health score based on P/E ratio and change
            pe_score = max(0, 100 - (stock['pe_ratio'] - 20) * 2)
            change_score = 50 + (stock['change'] * 5)
            health_score = int((pe_score + change_score) / 2)
            health_score = max(0, min(100, health_score))
            
            rating = "Excellent" if health_score >= 80 else "Good" if health_score >= 60 else "Fair" if health_score >= 40 else "Poor"
            return f"Financial Health Score: {health_score}/100 ({rating})\nFactors: P/E Ratio: {stock['pe_ratio']}, Recent Change: {stock['change']:+.1f}%"
        return f"Cannot calculate health for unknown symbol '{symbol}'"
    
    else:
        return f"Unknown tool: {name}"

@app.get("/")
async def health():
    return {"status": "healthy", "server": "Financial Data MCP Server", "symbols": list(MOCK_STOCKS.keys())}

@app.post("/mcp")
async def mcp_handler(request: JsonRpcRequest):

    # Handle notifications (return empty response)
    if request.method.startswith("notifications/"):
        return Response(status_code=204)
    
    response = JsonRpcResponse(id=request.id)
    
    if request.method == "initialize":
        response.result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "Financial Data MCP Server", "version": "1.0.0"}
        }
    
    elif request.method == "tools/list":
        response.result = {"tools": TOOLS}
    
    elif request.method == "tools/call":
        if not request.params:
            response.result = {"content": [{"type": "text", "text": "Missing parameters"}]}
        else:
            tool_name = request.params.get("name")
            arguments = request.params.get("arguments", {})
            result = execute_tool(tool_name, arguments)
            response.result = {"content": [{"type": "text", "text": result}]}
    
    else:
        response.result = {"error": {"code": -32601, "message": f"Method not found: {request.method}"}}
    
    return response.model_dump(exclude_none=True)

if __name__ == "__main__":
    print("🚀 Starting Financial Data MCP Server on port 8001...")
    print(f"📊 Available symbols: {', '.join(MOCK_STOCKS.keys())}")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="error")
