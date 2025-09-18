
import json
from datetime import datetime, timedelta
from fastmcp import FastMCP
import random
import math

# Initialize FastMCP server
mcp = FastMCP("analytics-reporting-server")

@mcp.tool()
def create_comparison_chart(companies: str, metric_type: str = "revenue") -> dict:
    """Create comparative analysis chart for multiple companies"""
    try:
        company_list = [c.strip().upper() for c in companies.split(",")]
        
        # Simulate chart data generation
        chart_data = {}
        for company in company_list:
            if metric_type.lower() == "revenue":
                base_value = random.randint(50, 500)  # Billions
                chart_data[company] = {
                    "current": base_value,
                    "previous": base_value * random.uniform(0.85, 1.15),
                    "growth_rate": round(random.uniform(-5, 15), 1)
                }
            elif metric_type.lower() == "pe_ratio":
                chart_data[company] = {
                    "current": round(random.uniform(15, 45), 1),
                    "industry_avg": 28.5,
                    "rating": random.choice(["Undervalued", "Fair", "Overvalued"])
                }
        
        # Generate insights
        best_performer = max(chart_data.items(), key=lambda x: x[1].get("growth_rate", 0))
        
        return {
            "chart_type": f"{metric_type.title()} Comparison",
            "companies_analyzed": len(company_list),
            "data": chart_data,
            "insights": {
                "best_performer": best_performer[0],
                "best_value": best_performer[1].get("growth_rate", "N/A"),
                "recommendation": f"{best_performer[0]} shows strongest {metric_type} performance"
            },
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Failed to create comparison chart: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }

@mcp.tool()
def calculate_portfolio_risk(holdings: str, investment_amount: float = 100000) -> dict:
    """Calculate portfolio risk assessment and recommendations"""
    return {
        "risk_score": 6.5,
        "risk_level": "Medium",
        "diversification_score": 8.2,
        "recommendations": ["Consider adding international exposure", "Monitor tech sector concentration"],
        "timestamp": datetime.now().isoformat()
    }

@mcp.tool()
def generate_financial_report(company: str, analysis_data: str = "") -> dict:
    """Generate comprehensive financial report with executive summary"""
    try:
        company = company.upper()
        
        # Parse any provided analysis data
        insights = []
        if analysis_data:
            if "positive" in analysis_data.lower():
                insights.append("Positive sentiment indicators detected")
            if "revenue" in analysis_data.lower():
                insights.append("Revenue metrics identified in analysis")
            if "risk" in analysis_data.lower():
                insights.append("Risk factors noted in assessment")
        
        # Generate report sections
        report = {
            "executive_summary": {
                "company": company,
                "report_date": datetime.now().strftime("%Y-%m-%d"),
                "overall_rating": random.choice(["Strong Buy", "Buy", "Hold", "Sell"]),
                "key_highlights": insights if insights else [
                    f"{company} shows stable fundamentals",
                    "Market position remains competitive",
                    "Financial metrics within industry norms"
                ]
            },
            "financial_overview": {
                "estimated_market_cap": f"${random.randint(100, 3000)}B",
                "sector": random.choice(["Technology", "Healthcare", "Finance", "Consumer"]),
                "analyst_consensus": random.choice(["Bullish", "Neutral", "Bearish"])
            },
            "risk_assessment": {
                "market_risk": random.choice(["Low", "Moderate", "High"]),
                "sector_risk": random.choice(["Low", "Moderate", "High"]),
                "company_specific_risk": random.choice(["Low", "Moderate", "High"])
            },
            "recommendations": {
                "investment_horizon": "12-18 months",
                "position_size": "5-10% of portfolio",
                "key_catalysts": [
                    "Quarterly earnings release",
                    "Product development updates", 
                    "Market expansion plans"
                ]
            },
            "report_metadata": {
                "generated_by": "Financial Intelligence Agent",
                "confidence_level": "Medium-High",
                "data_sources": "Multiple MCP servers",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return report
        
    except Exception as e:
        return {
            "error": f"Failed to generate financial report: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@mcp.tool()
def create_trend_analysis(symbol: str, timeframe: str = "12M") -> dict:
    """Create trend analysis and basic forecasting"""
    try:
        symbol = symbol.upper()
        
        # Generate simulated historical trend data
        periods = 12 if "12M" in timeframe else 24 if "24M" in timeframe else 6
        trend_data = []
        base_price = random.uniform(50, 300)
        
        for i in range(periods):
            # Simulate price movement with trend and volatility
            trend_factor = 1 + (0.02 * i / periods)  # Slight upward trend
            volatility = random.uniform(0.85, 1.15)
            price = base_price * trend_factor * volatility
            
            trend_data.append({
                "period": f"Month {i+1}",
                "price": round(price, 2),
                "volume_indicator": random.choice(["High", "Medium", "Low"])
            })
        
        # Calculate trend metrics
        start_price = trend_data[0]["price"]
        end_price = trend_data[-1]["price"]
        total_return = ((end_price - start_price) / start_price) * 100
        
        # Trend direction
        if total_return > 10:
            trend_direction = "Strong Uptrend"
        elif total_return > 0:
            trend_direction = "Mild Uptrend" 
        elif total_return > -10:
            trend_direction = "Sideways/Neutral"
        else:
            trend_direction = "Downtrend"
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "trend_data": trend_data,
            "analysis": {
                "trend_direction": trend_direction,
                "total_return": round(total_return, 2),
                "volatility": "Medium",  # Simplified
                "support_level": round(min(d["price"] for d in trend_data), 2),
                "resistance_level": round(max(d["price"] for d in trend_data), 2)
            },
            "forecast": {
                "next_month_outlook": random.choice(["Bullish", "Neutral", "Bearish"]),
                "confidence": random.choice(["High", "Medium", "Low"]),
                "key_factors": ["Market sentiment", "Sector performance", "Company fundamentals"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": f"Failed to create trend analysis: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("📊 Starting Analytics & Reporting MCP Server on port 8003...")
    print("🔧 Available tools: comparison charts, portfolio risk, financial reports, trend analysis")
    mcp.run(transport="http", host="127.0.0.1", port=8003)
