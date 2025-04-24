import argparse
import asyncio
import logging
import os
import json
import hmac
import hashlib
from datetime import datetime

import aiohttp
import websockets
import pandas as pd
import pandas_ta as ta
from ib_insync import IB, Stock, util

# --- CLI Arguments ---
parser = argparse.ArgumentParser(description="Alligator Trader Bot (ZeroHash execution)")
parser.add_argument("--primary-tf",  dest="primary_tf",   default="4 hours")
parser.add_argument("--longterm-tf", dest="longterm_tf",  default="8 hours")
args = parser.parse_args()

# --- ZeroHash Configuration (use ENV vars) ---
ZH_API_KEY    = os.getenv("ZH_API_KEY")
ZH_API_SECRET = os.getenv("ZH_API_SECRET")
ZH_BASE_URL   = "https://api.zerohash.com"
ZH_WS_URL     = "wss://ws.zerohash.com"
SYMBOL        = "BTC/USD"
ORDER_RISK_PCT = 0.01

# --- IB Configuration for data ---
IB_HOST = '127.0.0.1'
IB_PORT = 7497
IB_CLIENT_ID = 2
MAX_DATA_BARS = 300

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- ZeroHash REST & WS Client ---
class ZeroHashClient:
    def __init__(self, api_key, api_secret, base_url, ws_url):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.base_url   = base_url
        self.ws_url     = ws_url
        self.session    = aiohttp.ClientSession()
        self.ws         = None
        self.active_oco = {}  # {client_trade_id: {sl_id, tp_id}}

    async def _sign_headers(self, method, path, body=None):
        ts = str(int(datetime.utcnow().timestamp() * 1000))
        payload = ts + method.upper() + path + (json.dumps(body) if body else "")
        signature = hmac.new(self.api_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        return {
            "X-API-KEY":       self.api_key,
            "X-API-SIGNATURE": signature,
            "X-API-TIMESTAMP": ts,
            "Content-Type":    "application/json"
        }

    async def connect_ws(self):
        self.ws = await websockets.connect(self.ws_url)
        await self.ws.send(json.dumps({"type":"login","api_key":self.api_key}))
        asyncio.create_task(self._listen_ws())

    async def _listen_ws(self):
        async for msg in self.ws:
            data = json.loads(msg)
            if data.get("type") == "trade_capture_report" and data.get("status") == "filled":
                finished = data["order_id"]
                parent = data.get("client_trade_id")
                if parent in self.active_oco:
                    for oid in self.active_oco[parent]:
                        if oid != finished:
                            await self.cancel_order(oid)
                    del self.active_oco[parent]

    async def get_usd_balance(self):
        path = "/balances"
        headers = await self._sign_headers("GET", path)
        async with self.session.get(self.base_url + path, headers=headers) as resp:
            resp.raise_for_status()
            balances = (await resp.json()).get("balances", [])
            for bal in balances:
                if bal.get("asset") == "USD":
                    return float(bal.get("available", 0))
            return 0.0

    async def place_bracket_order(self, side, qty, stop_price, take_price):
        # Unique client_trade_id for OCO
        ct_id = f"alligator_{int(datetime.utcnow().timestamp()*1000)}"
        # 1) Market Entry
        p = {"symbol": SYMBOL, "side": side.lower(), "quantity": str(qty), "type": "market", "client_trade_id": ct_id}
        path = "/trades"
        h = await self._sign_headers("POST", path, p)
        async with self.session.post(self.base_url+path, json=p, headers=h) as r:
            r.raise_for_status()
            logger.info(f"Market {side} sent, id={ct_id}")
        # 2) STOP-LOSS
        sl = {"symbol": SYMBOL, "side":("sell" if side=="BUY" else "buy"),"type":"stop","quantity":str(qty),"stop_price":str(stop_price),"client_order_id":ct_id+"_SL"}
        path = "/orders"
        h2 = await self._sign_headers("POST", path, sl)
        async with self.session.post(self.base_url+path, json=sl, headers=h2) as r2:
            r2.raise_for_status(); sid=(await r2.json())["order_id"]
            logger.info(f"SL placed: {sid}")
        # 3) TAKE-PROFIT
        tp = {"symbol": SYMBOL, "side":("sell" if side=="BUY" else "buy"),"type":"limit","quantity":str(qty),"price":str(take_price),"client_order_id":ct_id+"_TP"}
        h3 = await self._sign_headers("POST", path, tp)
        async with self.session.post(self.base_url+path, json=tp, headers=h3) as r3:
            r3.raise_for_status(); tid=(await r3.json())["order_id"]
            logger.info(f"TP placed: {tid}")
        self.active_oco[ct_id] = {sid, tid}
        return ct_id

    async def cancel_order(self, oid):
        path = f"/orders/{oid}/cancel"
        h = await self._sign_headers("POST", path)
        async with self.session.post(self.base_url+path, headers=h) as r:
            logger.info(f"Canceled {oid}: {r.status}")

    async def close(self):
        await self.session.close()
        if self.ws: await self.ws.close()

# --- Indicator/IB Helpers ---
def timeframe_to_seconds(tf):
    val, unit = tf.split(); v=int(val); u=unit.lower()
    if 'min' in u: return v*60
    if 'hour' in u: return v*3600
    if 'day' in u: return v*86400
    raise ValueError(u)

async def get_historical_data(ib, contract, tf, duration):
    bars = await ib.reqHistoricalDataAsync(contract, '', f"{duration} D", tf, 'TRADES', False, 1)
    df = util.df(bars); df['date']=pd.to_datetime(df['date']); df.set_index('date',inplace=True)
    return df

def calculate_indicators(df, label):
    ms = max(14, 13, 8, 5, 100)
    if len(df)<ms: return pd.DataFrame()
    out=df.copy()
    out.ta.alligator(jaw_length=13,teeth_length=8,lips_length=5,append=True,col_names=(f'jaw_{label}',f'teeth_{label}',f'lips_{label}'))
    if label=='primary':
        out.ta.cmo(length=14,append=True); out.rename(columns={'CMO_14':'cmo'},inplace=True)
        out.ta.stochrsi(length=14,rsi_length=14,k=3,d=3,append=True)
        out.rename(columns={f'STOCHRSIk_14_14_3_3':'stoch_k',f'STOCHRSId_14_14_3_3':'stoch_d'},inplace=True)
        out.ta.atr(length=14,append=True); out.rename(columns={'ATRr_14':'atr'},inplace=True)
    else:
        out.ta.adx(length=14,append=True); out.rename(columns={'ADX_14':'adx'},inplace=True)
        out.ta.ema(length=100,append=True); out.rename(columns={'EMA_100':'ema100'},inplace=True)
    out.dropna(inplace=True); return out

def determine_trend(j, t, l):
    if l>t>j: return 'UP'
    if l<t<j: return 'DOWN'
    return 'NONE'

# --- Main Loop ---
async def run_strategy(ib, contract, zh):
    # 1) Data
    pri_df = await get_historical_data(ib, contract, args.primary_tf, MAX_DATA_BARS)
    long_df= await get_historical_data(ib, contract, args.longterm_tf, MAX_DATA_BARS)
    p_i=calculate_indicators(pri_df,'primary'); l_i=calculate_indicators(long_df,'longterm')
    if p_i.empty or l_i.empty: return
    latest_p = p_i.iloc[-1]; t0=latest_p.name
    aligned = l_i.loc[:t0]; latest_l = aligned.iloc[-1]
    price = latest_p['close']

    # 2) Signals
    trend_p = determine_trend(latest_p['jaw_primary'],latest_p['teeth_primary'],latest_p['lips_primary'])
    trend_l = determine_trend(latest_l['jaw_longterm'],latest_l['teeth_longterm'],latest_l['lips_longterm'])
    adv = (price>latest_l['ema100']); adx=latest_l['adx']; cmo=latest_p['cmo']; stoch=latest_p['stoch_k']
    side=None
    if trend_p=='UP' and trend_l=='UP' and adv and adx>25 and cmo>5 and stoch<20: side='BUY'
    if trend_p=='DOWN' and trend_l=='DOWN' and not adv and adx>25 and cmo< -5 and stoch>80: side='SELL'
    if not side: return

    # 3) Sizing
    atr=latest_p['atr']; sl_price = price - atr*1.5 if side=='BUY' else price+atr*1.5
    tp_price = price + atr*3.0 if side=='BUY' else price-atr*3.0
    bal = await zh.get_usd_balance(); risk = bal*ORDER_RISK_PCT; qty = max(1, int(risk/(abs(price-sl_price))))

    # 4) Execute
    ct = await zh.place_bracket_order(side, qty, sl_price, tp_price)
    logger.info(f"Bracket order sent: {ct}")

async def main():
    # IB connect
    ib = IB(); await ib.connectAsync(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
    contract = (await ib.qualifyContractsAsync(Stock('BTC','SMART','USD')))[0]
    # ZeroHash connect
    zh = ZeroHashClient(ZH_API_KEY, ZH_API_SECRET, ZH_BASE_URL, ZH_WS_URL)
    await zh.connect_ws()

    while True:
        try:
            await run_strategy(ib, contract, zh)
        except Exception as e:
            logger.exception(e)
        await asyncio.sleep(timeframe_to_seconds(args.primary_tf))

    await zh.close()

if __name__=='__main__':
    asyncio.run(main())
