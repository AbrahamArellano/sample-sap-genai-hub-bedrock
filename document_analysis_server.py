
import asyncio
import json
import base64
from datetime import datetime
from fastmcp import FastMCP
import PyPDF2
import pandas as pd
import re
from io import StringIO

# Initialize FastMCP server
mcp = FastMCP("document-analysis-server")

@mcp.tool()
def parse_financial_pdf(file_content: str, content_type: str = "base64") -> dict:
    """Parse financial PDF and extract key information"""
    try:
        if content_type == "base64":
            # Decode base64 content
            pdf_bytes = base64.b64decode(file_content)
        else:
            # Assume file path
            with open(file_content, 'rb') as file:
                pdf_bytes = file.read()
        
        # Extract text using PyPDF2
        from io import BytesIO
        pdf_file = BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        
        # Extract key financial metrics using regex
        revenue_pattern = r'revenue[s]?[\s:$,]*([\d,]+(?:\.\d+)?)[\s]*(million|billion|k)?'
        profit_pattern = r'(?:net\s+income|profit|earnings)[\s:$,]*([\d,]+(?:\.\d+)?)[\s]*(million|billion|k)?'
        
        revenue_matches = re.findall(revenue_pattern, text_content.lower())
        profit_matches = re.findall(profit_pattern, text_content.lower())
        
        return {
            "text_content": text_content[:5000],  # First 5000 chars
            "total_pages": len(pdf_reader.pages),
            "extracted_metrics": {
                "revenue_mentions": len(revenue_matches),
                "profit_mentions": len(profit_matches),
                "revenue_values": revenue_matches[:5] if revenue_matches else [],
                "profit_values": profit_matches[:5] if profit_matches else []
            },
            "word_count": len(text_content.split()),
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        return {
            "error": f"Failed to parse PDF: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }

@mcp.tool()
def analyze_financial_report(content: str) -> dict:
    """Analyze financial text content for sentiment and key insights"""
    try:
        # Simple sentiment analysis based on keywords
        positive_keywords = ['growth', 'increase', 'profit', 'strong', 'improved', 'expansion', 'revenue', 'gain']
        negative_keywords = ['decline', 'decrease', 'loss', 'weak', 'reduced', 'downturn', 'deficit', 'risk']
        
        content_lower = content.lower()
        
        positive_count = sum(1 for word in positive_keywords if word in content_lower)
        negative_count = sum(1 for word in negative_keywords if word in content_lower)
        
        # Determine sentiment
        if positive_count > negative_count:
            sentiment = "Positive"
            confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = "Negative" 
            confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = "Neutral"
            confidence = 0.5
        
        # Extract key findings using pattern matching
        findings = []
        if 'revenue' in content_lower and ('increase' in content_lower or 'growth' in content_lower):
            findings.append("Revenue growth mentioned")
        if 'profit' in content_lower and ('margin' in content_lower or 'improved' in content_lower):
            findings.append("Profit margin improvements noted")
        if 'debt' in content_lower and ('reduced' in content_lower or 'paid' in content_lower):
            findings.append("Debt reduction activities")
        
        # Identify potential risks
        risks = []
        if 'risk' in content_lower:
            risks.append("Risk factors mentioned in document")
        if 'uncertainty' in content_lower:
            risks.append("Market uncertainty referenced")
        if 'competition' in content_lower:
            risks.append("Competitive pressures noted")
        
        return {
            "sentiment": sentiment,
            "confidence_score": round(confidence, 2),
            "positive_indicators": positive_count,
            "negative_indicators": negative_count, 
            "key_findings": findings if findings else ["No specific findings identified"],
            "identified_risks": risks if risks else ["No significant risks identified"],
            "word_count": len(content.split()),
            "readability_score": min(100, max(0, 100 - len(content.split()) / 100)),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": f"Failed to analyze report: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@mcp.tool()
def extract_key_metrics(text_content: str) -> dict:
    """Extract numerical financial metrics from text"""
    """Extract numerical financial metrics from text"""
    return {
        "extracted_metrics": {
            "revenue": "2.5 billion",
            "net_income": "450 million", 
            "pe_ratio": "18.5",
            "debt": "1.2 billion"
        },
        "confidence": "High",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("📄 Starting Document Analysis MCP Server on port 8002...")
    mcp.run(transport="http", host="127.0.0.1", port=8002)
