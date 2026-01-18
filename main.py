import yfinance as yf
import pandas_ta as ta
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 1. Configuration de Gemini via LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2 # On garde une température basse pour la précision technique
)

def analyze_with_langchain(data_summary):
    # 2. Définition du Template (La Stratégie)
    template = """
    Tu es un expert en trading algorithmique spécialisé dans le marché Forex.
    Voici les données techniques récentes pour l'actif {ticker} :
    
    - Prix de clôture : {price}
    - RSI (Indicateur de force) : {rsi}
    - Moyenne Mobile 20 : {sma20}
    - Moyenne Mobile 50 : {sma50}
    
    Analyse de stratégie :
    1. Si le prix est au-dessus des deux SMA et que le RSI est < 70, la tendance est haussière saine.
    2. Si le prix est en dessous des deux SMA et que le RSI est > 30, la tendance est baissière saine.
    3. Si le RSI est > 70 ou < 30, il y a un risque de retournement imminent.
    
    Donne ta recommandation finale sous ce format :
    SIGNAL: [ACHAT/VENTE/ATTENTE]
    LOGIQUE: [Ta justification courte]
    STOP_LOSS: [Calcul un niveau de prix logique]
    """
    
    prompt = PromptTemplate(
        input_variables=["ticker", "price", "rsi", "sma20", "sma50"],
        template=template
    )
    
    # 3. Création de la chaîne
    chain = prompt | llm
    
    response = chain.invoke(data_summary)
    return response.content

def main():
    ticker_symbol = "EURUSD=X"
    df = yf.Ticker(ticker_symbol).history(period="3mo")
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    last = df.iloc[-1]
    
    # Préparation des données pour LangChain
    data_for_ai = {
        "ticker": ticker_symbol,
        "price": round(last['Close'], 4),
        "rsi": round(last['RSI'], 2),
        "sma20": round(last['SMA_20'], 4),
        "sma50": round(last['SMA_50'], 4)
    }
    
    result = analyze_with_langchain(data_for_ai)
    print(result)

if __name__ == "__main__":
    main()