import os
import requests
import yfinance as yf
import pandas_ta as ta
import json
from typing import TypedDict, Annotated, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
# Assure-toi d'avoir GOOGLE_API_KEY et TAVILY_API_KEY dans ton .env
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
tavily_tool = TavilySearchResults(max_results=3)

# --- 1. DÉFINITION DE L'ÉTAT DU GRAPHE ---
class AgentState(TypedDict):
    ticker: str
    symbol_name: str # ex: EURUSD
    technical_data: dict
    news_data: str
    web_strategy_data: str
    final_report: str

# --- 2. LES OUTILS (NODES) ---

def fetch_technicals_node(state: AgentState):
    """Calcule les indicateurs techniques (Hard Data)"""
    ticker = state["ticker"]
    print(f"📊 [1/4] Analyse technique sur {ticker}...")
    
    df = yf.Ticker(ticker).history(period="3mo")
    if df.empty:
        return {"technical_data": {"error": "No data found"}}

    # Calculs
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    tech_summary = {
        "price": round(last['Close'], 5),
        "rsi": round(last['RSI'], 2),
        "sma20": round(last['SMA_20'], 5),
        "sma50": round(last['SMA_50'], 5),
        "trend_sma": "BULLISH" if last['SMA_20'] > last['SMA_50'] else "BEARISH",
        "previous_close": round(prev['Close'], 5)
    }
    return {"technical_data": tech_summary}

def fetch_market_news_node(state: AgentState):
    """Récupère les news via l'API TradingView fournie"""
    symbol = state["symbol_name"] # ex: EURUSD
    print(f"📰 [2/4] Recherche de news pour {symbol}...")
    
    # URL adaptée dynamiquement ou fixe selon tes besoins
    url = f"https://news-mediator.tradingview.com/public/view/v1/symbol?filter=lang%3Afr&filter=symbol%3ATICKMILL%3A{symbol}&client=web&user_prostatus=non_pro"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # On extrait les titres pertinents (les 5 premiers)
            news_items = [item['title'] for item in data[:5]]
            news_summary = "\n- ".join(news_items)
        else:
            news_summary = "No news accessible via API."
    except Exception as e:
        news_summary = f"Error fetching news: {str(e)}"
        
    return {"news_data": news_summary}

def fetch_web_strategies_node(state: AgentState):
    """Cherche des stratégies actuelles sur le web via Tavily"""
    symbol = state["symbol_name"]
    print(f"🌐 [3/4] Recherche de stratégies web pour {symbol}...")
    
    query = f"best trading strategy for {symbol} current market conditions 2025 2026 forecast analysis"
    try:
        results = tavily_tool.invoke(query)
        # On condense les résultats
        strategies = "\n".join([f"Source: {res['url']}\nContent: {res['content'][:300]}..." for res in results])
    except Exception as e:
        strategies = "Could not fetch web strategies."
        
    return {"web_strategy_data": strategies}

def strategist_agent_node(state: AgentState):
    """Le cerveau : Synthétise tout et génère la décision"""
    print(f"🧠 [4/4] Synthèse et Prise de décision...")
    
    # SYSTEM PROMPT EN ANGLAIS (Pour meilleure performance logique)
    system_prompt = """
    You are a Senior Hedge Fund Portfolio Manager. Your goal is to make a high-stakes trading decision based on multiple data sources.
    
    You have access to 3 types of data:
    1. **Technical Analysis (Hard Data):** RSI, Moving Averages.
    2. **Market News (Sentiment):** Real-time headlines affecting the asset.
    3. **Web Strategies (Smart Money):** What other analysts are currently discussing.

    YOUR TASK:
    Analyze the correlation between the Technicals and the News. 
    - If Technicals say BUY but News is very negative (e.g., War, Tariffs), you must be CAUTIOUS or SELL.
    - If both align, the signal is STRONG.

    OUTPUT FORMAT (in French language for the final user):
    Returns a structured report in Markdown:
    
    # 📢 DÉCISION FINALE : [ACHAT / VENTE / ATTENTE] (en Gras et Majuscules)
    
    ## 1. Synthèse de la Situation
    Explain the context. Is the market driven by fear (news) or greed? clash between technicals and news?
    
    ## 2. Arguments Clés
    * **Technique :** [Analyze RSI/SMA levels]
    * **Fondamental & News :** [Analyze the impact of the fetched news]
    * **Sentiment Web :** [What are the blogs/analysts saying?]
    
    ## 3. Stratégie Recommandée
    Propose a concrete strategy (e.g., "Wait for pullback to X", "Enter now with tight stop").
    * **Entry Price:** ...
    * **Stop Loss:** ... (Calculate a logical level based on data)
    * **Take Profit:** ...
    
    ## 4. Why this strategy?
    Give a solid argument why this specific plan minimizes risk.
    """

    user_content = f"""
    ASSET: {state['symbol_name']} ({state['ticker']})
    
    --- TECHNICAL DATA ---
    {json.dumps(state['technical_data'])}
    
    --- BREAKING NEWS ---
    {state['news_data']}
    
    --- WEB STRATEGIES ---
    {state['web_strategy_data']}
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ]
    
    response = llm.invoke(messages)
    return {"final_report": response.content}

# --- 3. CONSTRUCTION DU GRAPHE (LANGGRAPH) ---

workflow = StateGraph(AgentState)

# Ajout des noeuds
workflow.add_node("get_technicals", fetch_technicals_node)
workflow.add_node("get_news", fetch_market_news_node)
workflow.add_node("get_web_strategies", fetch_web_strategies_node)
workflow.add_node("decision_maker", strategist_agent_node)

# Définition du flux
# On commence par récupérer les données (on pourrait le faire en parallèle, ici en série pour simplifier)
workflow.set_entry_point("get_technicals")
workflow.add_edge("get_technicals", "get_news")
workflow.add_edge("get_news", "get_web_strategies")
workflow.add_edge("get_web_strategies", "decision_maker")
workflow.add_edge("decision_maker", END)

# Compilation
app = workflow.compile()

# --- 4. EXECUTION ---

def main():
    # Exemple : EURUSD
    initial_state = {
        "ticker": "EURUSD=X",
        "symbol_name": "EURUSD",
        "technical_data": {},
        "news_data": "",
        "web_strategy_data": "",
        "final_report": ""
    }
    
    print("🚀 Démarrage de l'Agent de Trading AI...")
    result = app.invoke(initial_state)
    
    print("\n" + "="*50 + "\n")
    print(result["final_report"])
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()