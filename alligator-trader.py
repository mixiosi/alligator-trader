import asyncio
import logging
from datetime import datetime, timedelta
import time

import pandas as pd
import pandas_ta as ta
from ib_insync import IB, Stock, Contract, MarketOrder, BracketOrder, util

# --- Configuration ---
# TWS/Gateway Connection
IB_HOST = '127.0.0.1'
IB_PORT = 7497  # 7497 for TWS Paper, 7496 for TWS Live, 4002 for Gateway Paper, 4001 for Gateway Live
IB_CLIENT_ID = 1 # Choose a unique client ID

# Trading Parameters
SYMBOL = 'MSTR'
EXCHANGE = 'SMART'
CURRENCY = 'USD'
PRIMARY_TIMEFRAME = '4 hours' # Timeframe for primary signals (CMO, StochRSI, current Alligator, ATR)
LONGTERM_TIMEFRAME = '6 hours' # Timeframe for long-term confirmation (ADX, LT Alligator, LT EMA)
ORDER_SIZE_PERCENT = 0.02 # Risk 2% of portfolio value per trade (adjust as needed)
SL_ATR_MULTIPLIER = 1.5   # Stop loss distance in ATR multiples
TP_ATR_MULTIPLIER = 3.0   # Take profit distance in ATR multiples

# Indicator Settings
ADX_PERIOD = 14
CMO_PERIOD = 14
STOCHRSI_PERIOD = 14
STOCHRSI_K = 3
STOCHRSI_D = 3
ALLIGATOR_JAW = 13
ALLIGATOR_TEETH = 8
ALLIGATOR_LIPS = 5
EMA_LONGTERM_PERIOD = 100
ATR_PERIOD = 14 # Used for SL/TP calculation on PRIMARY_TIMEFRAME

# Strategy Thresholds
ADX_THRESHOLD = 25       # Minimum ADX to confirm trend strength
CMO_LONG_THRESHOLD = 5   # CMO must be above this for long entry
CMO_SHORT_THRESHOLD = -5  # CMO must be below this for short entry
STOCHRSI_OVERSOLD = 20   # StochRSI below this indicates oversold (for long)
STOCHRSI_OVERBOUGHT = 80 # StochRSI above this indicates overbought (for short)

# Other Settings
RUN_INTERVAL_SECONDS = 4 * 60 * 60 # Check every 4 hours
# RUN_INTERVAL_SECONDS = 60 # For testing: Check every minute
MAX_DATA_REQUEST_BARS = 300 # How many bars to fetch (ensure enough for longest lookback)
LOG_LEVEL = logging.INFO # DEBUG, INFO, WARNING, ERROR

# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
ib = None # IB connection object
contract = None # Stock contract object

# --- Helper Functions ---

def create_contract(symbol, exchange, currency):
    """Creates an IB stock contract object."""
    return Stock(symbol, exchange, currency)

async def get_historical_data(ib_conn: IB, contract: Contract, timeframe: str, duration: str):
    """Fetches historical OHLCV data."""
    try:
        logger.debug(f"Requesting historical data for {contract.symbol} - Timeframe: {timeframe}, Duration: {duration}")
        bars = await ib_conn.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=timeframe,
            whatToShow='TRADES',
            useRTH=False, # Use data outside regular trading hours if available
            formatDate=1 # Return as datetime objects
        )
        if not bars:
            logger.warning(f"No historical data returned for {contract.symbol} - {timeframe}")
            return pd.DataFrame()

        df = util.df(bars)
        if df is None or df.empty:
             logger.warning(f"Empty DataFrame after conversion for {contract.symbol} - {timeframe}")
             return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        logger.info(f"Successfully retrieved {len(df)} bars for {contract.symbol} - {timeframe}")
        return df[['open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        logger.error(f"Error fetching historical data for {contract.symbol} - {timeframe}: {e}")
        return pd.DataFrame()

def calculate_indicators(df: pd.DataFrame, timeframe_label: str):
    """Calculates and adds technical indicators to the DataFrame."""
    if df.empty:
        logger.warning(f"Cannot calculate indicators on empty DataFrame ({timeframe_label})")
        return df

    logger.debug(f"Calculating indicators for {timeframe_label} timeframe")
    try:
        # Alligator
        df.ta.alligator(jaw_length=ALLIGATOR_JAW, teeth_length=ALLIGATOR_TEETH, lips_length=ALLIGATOR_LIPS, append=True)
        df.rename(columns={
            f'ALLIGATORj_{ALLIGATOR_JAW}_{ALLIGATOR_TEETH}_{ALLIGATOR_LIPS}': f'alligator_jaw_{timeframe_label}',
            f'ALLIGATORt_{ALLIGATOR_JAW}_{ALLIGATOR_TEETH}_{ALLIGATOR_LIPS}': f'alligator_teeth_{timeframe_label}',
            f'ALLIGATORl_{ALLIGATOR_JAW}_{ALLIGATOR_TEETH}_{ALLIGATOR_LIPS}': f'alligator_lips_{timeframe_label}'
        }, inplace=True)

        # ADX (only needed for long-term timeframe)
        if timeframe_label == 'longterm':
            df.ta.adx(length=ADX_PERIOD, append=True)
            df.rename(columns={f'ADX_{ADX_PERIOD}': 'adx'}, inplace=True)

        # CMO (only needed for primary timeframe)
        if timeframe_label == 'primary':
            df.ta.cmo(length=CMO_PERIOD, append=True)
            df.rename(columns={f'CMO_{CMO_PERIOD}': 'cmo'}, inplace=True)

        # Stochastic RSI (only needed for primary timeframe)
        if timeframe_label == 'primary':
            stoch_rsi = df.ta.stochrsi(length=STOCHRSI_PERIOD, rsi_length=STOCHRSI_PERIOD, k=STOCHRSI_K, d=STOCHRSI_D, append=True)
            df.rename(columns={
                 f'STOCHRSIk_{STOCHRSI_PERIOD}_{STOCHRSI_PERIOD}_{STOCHRSI_K}_{STOCHRSI_D}': 'stochrsi_k',
                 f'STOCHRSId_{STOCHRSI_PERIOD}_{STOCHRSI_PERIOD}_{STOCHRSI_K}_{STOCHRSI_D}': 'stochrsi_d'
                 }, inplace=True)


        # EMA (only needed for long-term timeframe)
        if timeframe_label == 'longterm':
             df.ta.ema(length=EMA_LONGTERM_PERIOD, append=True)
             df.rename(columns={f'EMA_{EMA_LONGTERM_PERIOD}': f'ema_{EMA_LONGTERM_PERIOD}'}, inplace=True)

        # ATR (only needed for primary timeframe, for SL/TP)
        if timeframe_label == 'primary':
             df.ta.atr(length=ATR_PERIOD, append=True)
             df.rename(columns={f'ATRr_{ATR_PERIOD}': 'atr'}, inplace=True)

        df.dropna(inplace=True) # Remove rows with NaN indicators
        logger.debug(f"Finished calculating indicators for {timeframe_label}")

    except Exception as e:
        logger.error(f"Error calculating indicators for {timeframe_label}: {e}")
        # Return original df potentially without all indicators if error occurred mid-way
    return df


def determine_alligator_trend(jaw, teeth, lips):
    """Determines trend based on the latest Alligator values."""
    if pd.isna(jaw) or pd.isna(teeth) or pd.isna(lips):
        return "UNKNOWN" # Not enough data
    if lips > teeth > jaw:
        return "UPTREND"
    elif lips < teeth < jaw:
        return "DOWNTREND"
    else:
        return "NO TREND" # Ranging or unclear

async def get_account_value(ib_conn: IB, currency: str = 'USD'):
    """Retrieves the NetLiquidation value for the account."""
    acc_summary = await ib_conn.reqAccountSummaryAsync(account='All', tags='NetLiquidation')
    if not acc_summary:
        logger.error("Could not retrieve account summary.")
        return None

    for item in acc_summary:
        if item.currency == currency and item.tag == 'NetLiquidation':
            logger.info(f"Account Net Liquidation ({currency}): {item.value}")
            try:
                return float(item.value)
            except ValueError:
                logger.error(f"Could not convert account value '{item.value}' to float.")
                return None
    logger.warning(f"NetLiquidation value not found for currency {currency}.")
    return None

async def check_existing_position(ib_conn: IB, contract: Contract):
    """Checks if there is an existing position for the contract."""
    positions = await ib_conn.reqPositionsAsync()
    for pos in positions:
        if pos.contract.conId == contract.conId:
            logger.info(f"Existing position found for {contract.symbol}: {pos.position} shares.")
            return pos.position # Returns size (positive for long, negative for short)
    logger.info(f"No existing position found for {contract.symbol}.")
    return 0 # No position


async def place_bracket_order(ib_conn: IB, contract: Contract, action: str, quantity: float, limit_price: float, stop_loss_price: float, take_profit_price: float):
    """Places a bracket order (Market entry with SL and TP)."""
    if action not in ['BUY', 'SELL']:
        logger.error(f"Invalid action for order: {action}")
        return None

    # Round prices to appropriate tick size (simplistic rounding here, IB might require specific rules)
    stop_loss_price = round(stop_loss_price, 2)
    take_profit_price = round(take_profit_price, 2)

    # Create base market order
    base_order = MarketOrder(action=action, totalQuantity=quantity)
    # base_order.transmit = False # Important: Don't transmit base order alone

    # Create stop loss order
    sl_action = 'SELL' if action == 'BUY' else 'BUY'
    stop_loss_order = StopOrder(
        action=sl_action,
        totalQuantity=quantity,
        stopPrice=stop_loss_price,
        # parentId=base_order.orderId, # IBPy does this automatically
        # transmit=False
    )

    # Create take profit order
    tp_action = 'SELL' if action == 'BUY' else 'BUY'
    take_profit_order = LimitOrder(
        action=tp_action,
        totalQuantity=quantity,
        lmtPrice=take_profit_price,
        # parentId=base_order.orderId,
        # transmit=True # The last order in bracket transmits all
    )


    # Use BracketOrder for simplicity (ib_insync feature)
    bracket = BracketOrder(
        action=action,
        totalQuantity=quantity,
        limitPrice=limit_price, # This seems misnamed for Market entry in BracketOrder docs, check behavior - may need separate market + attached SL/TP
        takeProfitPrice=take_profit_price,
        stopLossPrice=stop_loss_price
    )

    # Use a simple market order and attach SL/TP for more control if BracketOrder with Market is tricky
    logger.info("Using Market Order with separate SL/TP attachments")
    base_mkt_order = MarketOrder(action, quantity, transmit=False) # Don't send yet
    sl_order = StopOrder('SELL' if action == 'BUY' else 'BUY', quantity, stop_loss_price, transmit=False, parentId=0) # ParentId will be set later
    tp_order = LimitOrder('SELL' if action == 'BUY' else 'BUY', quantity, take_profit_price, transmit=True, parentId=0) # Transmit last order

    try:
        # Qualify contract if not already done (recommended)
        [qual_contract] = await ib_conn.qualifyContractsAsync(contract)

        # Place orders
        trade = ib_conn.placeOrder(qual_contract, base_mkt_order)
        logger.info(f"Placing Base Market Order: {action} {quantity} {contract.symbol} - OrderId: {trade.order.orderId}")
        await trade.orderStatusEvent # Wait for orderId confirmation

        if trade.order.orderId:
             sl_order.parentId = trade.order.orderId
             tp_order.parentId = trade.order.orderId
             sl_trade = ib_conn.placeOrder(qual_contract, sl_order)
             tp_trade = ib_conn.placeOrder(qual_contract, tp_order)
             logger.info(f"Placed SL Order - OrderId: {sl_trade.order.orderId}, ParentId: {sl_order.parentId}")
             logger.info(f"Placed TP Order - OrderId: {tp_trade.order.orderId}, ParentId: {tp_order.parentId}")
             return trade, sl_trade, tp_trade # Return all trades
        else:
             logger.error(f"Failed to get OrderId for base market order. Cannot place SL/TP.")
             # Attempt to cancel the base order if possible (needs order tracking)
             return None

    except Exception as e:
        logger.error(f"Error placing {action} bracket order for {contract.symbol}: {e}")
        # Consider canceling partial orders if any were placed
        return None


# --- Main Strategy Logic ---

async def run_strategy():
    """Main loop to fetch data, check strategy conditions, and place orders."""
    global ib, contract

    if not ib or not ib.isConnected():
        logger.error("IB connection is not available.")
        return

    if not contract:
        logger.error("Contract is not defined.")
        return

    logger.info(f"--- Running Strategy Check for {SYMBOL} at {datetime.now()} ---")

    # --- 1. Check Existing Position ---
    current_position = await check_existing_position(ib, contract)
    if current_position != 0:
        logger.info(f"Already have a position of {current_position} for {SYMBOL}. Skipping new entry check.")
        return # Simple strategy: only enter if flat

    # --- 2. Get Account Value for Sizing ---
    account_value = await get_account_value(ib, CURRENCY)
    if account_value is None:
        logger.error("Could not get account value. Skipping trade.")
        return

    # --- 3. Get Historical Data ---
    # Calculate duration string needed - needs buffer for indicator lookbacks
    # Example: EMA100 needs 100 bars + ~50 for smoothing = 150. ADX ~30. Max is ~150 bars.
    # Fetch MAX_DATA_REQUEST_BARS bars for safety.
    duration_str = f"{MAX_DATA_REQUEST_BARS * 2} D" # Fetch more days to ensure enough bars even on longer TFs
    if PRIMARY_TIMEFRAME.endswith("hours"):
         duration_str = f"{max(MAX_DATA_REQUEST_BARS // (24 // int(PRIMARY_TIMEFRAME.split()[0])), 5)} D" # Min 5 days
    elif PRIMARY_TIMEFRAME.endswith("mins"):
         duration_str = f"{max(MAX_DATA_REQUEST_BARS // (24 * 60 // int(PRIMARY_TIMEFRAME.split()[0])), 2)} D" # Min 2 days

    df_primary = await get_historical_data(ib, contract, PRIMARY_TIMEFRAME, duration_str)
    if df_primary.empty:
        logger.warning(f"Could not get primary timeframe ({PRIMARY_TIMEFRAME}) data. Skipping.")
        return

    df_longterm = await get_historical_data(ib, contract, LONGTERM_TIMEFRAME, duration_str) # Use same duration fetch
    if df_longterm.empty:
        logger.warning(f"Could not get long-term timeframe ({LONGTERM_TIMEFRAME}) data. Skipping.")
        return

    # --- 4. Calculate Indicators ---
    df_primary = calculate_indicators(df_primary, 'primary')
    df_longterm = calculate_indicators(df_longterm, 'longterm')

    if df_primary.empty or df_longterm.empty:
        logger.warning("Indicator calculation resulted in empty data. Skipping check.")
        return

    # --- 5. Get Latest Data ---
    try:
        latest_primary = df_primary.iloc[-1]
        latest_longterm = df_longterm.iloc[-1]
        current_price = latest_primary['close'] # Use primary close price for checks & orders
    except IndexError:
        logger.warning("Not enough data rows after indicator calculation. Skipping check.")
        return

    # Check if all required indicators are present
    required_primary_cols = ['close', 'atr', 'cmo', 'stochrsi_k', 'stochrsi_d', 'alligator_jaw_primary', 'alligator_teeth_primary', 'alligator_lips_primary']
    required_longterm_cols = ['close', 'adx', 'ema_100', 'alligator_jaw_longterm', 'alligator_teeth_longterm', 'alligator_lips_longterm']

    if not all(col in latest_primary for col in required_primary_cols):
         logger.error(f"Missing required indicators in primary data: {latest_primary.index.tolist()}")
         return
    if not all(col in latest_longterm for col in required_longterm_cols):
         logger.error(f"Missing required indicators in long-term data: {latest_longterm.index.tolist()}")
         return

    # Check for NaN values in latest row (should have been dropped, but double check)
    if latest_primary.isnull().any() or latest_longterm.isnull().any():
         logger.warning(f"Latest data contains NaN values. Skipping check. Primary:\n{latest_primary[latest_primary.isnull()]}\nLongterm:\n{latest_longterm[latest_longterm.isnull()]}")
         return


    # --- 6. Evaluate Strategy Conditions ---
    primary_trend = determine_alligator_trend(latest_primary['alligator_jaw_primary'], latest_primary['alligator_teeth_primary'], latest_primary['alligator_lips_primary'])
    longterm_trend = determine_alligator_trend(latest_longterm['alligator_jaw_longterm'], latest_longterm['alligator_teeth_longterm'], latest_longterm['alligator_lips_longterm'])
    longterm_ema_trend_up = current_price > latest_longterm[f'ema_{EMA_LONGTERM_PERIOD}']

    logger.info(f"Latest Primary Data ({latest_primary.name}): Price={current_price:.2f}, Trend={primary_trend}, CMO={latest_primary['cmo']:.2f}, StochK={latest_primary['stochrsi_k']:.2f}, ATR={latest_primary['atr']:.2f}")
    logger.info(f"Latest LongTerm Data ({latest_longterm.name}): Trend={longterm_trend}, ADX={latest_longterm['adx']:.2f}, AboveEMA{EMA_LONGTERM_PERIOD}={longterm_ema_trend_up}")

    trade_action = None

    # --- Long Entry Conditions ---
    if (primary_trend == "UPTREND" and
        latest_longterm['adx'] > ADX_THRESHOLD and
        longterm_trend == "UPTREND" and # Check long-term Alligator trend
        longterm_ema_trend_up and       # Price above long-term EMA
        latest_primary['cmo'] > CMO_LONG_THRESHOLD and
        latest_primary['stochrsi_k'] < STOCHRSI_OVERSOLD): # Using Stoch K < oversold
        logger.info(">>> LONG ENTRY CONDITIONS MET <<<")
        trade_action = 'BUY'

    # --- Short Entry Conditions ---
    elif (primary_trend == "DOWNTREND" and
          latest_longterm['adx'] > ADX_THRESHOLD and
          longterm_trend == "DOWNTREND" and # Check long-term Alligator trend
          not longterm_ema_trend_up and     # Price below long-term EMA
          latest_primary['cmo'] < CMO_SHORT_THRESHOLD and
          latest_primary['stochrsi_k'] > STOCHRSI_OVERBOUGHT): # Using Stoch K > overbought
        logger.info(">>> SHORT ENTRY CONDITIONS MET <<<")
        trade_action = 'SELL'

    else:
        logger.info("Conditions not met for entry.")

    # --- 7. Place Order If Conditions Met ---
    if trade_action:
        atr_value = latest_primary['atr']
        if pd.isna(atr_value) or atr_value == 0:
            logger.error("ATR value is invalid. Cannot calculate SL/TP or position size.")
            return

        # Calculate Stop Loss and Take Profit
        if trade_action == 'BUY':
            stop_loss = current_price - (atr_value * SL_ATR_MULTIPLIER)
            take_profit = current_price + (atr_value * TP_ATR_MULTIPLIER)
            risk_per_share = current_price - stop_loss
        else: # SELL
            stop_loss = current_price + (atr_value * SL_ATR_MULTIPLIER)
            take_profit = current_price - (atr_value * TP_ATR_MULTIPLIER)
            risk_per_share = stop_loss - current_price

        if risk_per_share <= 0:
            logger.error(f"Risk per share is zero or negative ({risk_per_share:.2f}). Cannot calculate position size.")
            return

        # Calculate Position Size
        trade_value = account_value * ORDER_SIZE_PERCENT
        quantity = int(trade_value / risk_per_share) # Basic sizing based on risk % and stop distance

        if quantity <= 0:
            logger.warning(f"Calculated order quantity is {quantity}. Minimum is 1 share. Check risk % or account value.")
            return

        logger.info(f"Attempting to place {trade_action} order:")
        logger.info(f"  Quantity: {quantity}")
        logger.info(f"  Entry Price (Market): ~{current_price:.2f}")
        logger.info(f"  Stop Loss: {stop_loss:.2f}")
        logger.info(f"  Take Profit: {take_profit:.2f}")
        logger.info(f"  Risk per Share: {risk_per_share:.2f}")
        logger.info(f"  Estimated Trade Value: {quantity * current_price:.2f}")


        await place_bracket_order(ib, contract, trade_action, quantity, current_price, stop_loss, take_profit)


# --- Main Execution ---

async def main():
    global ib, contract
    ib = IB()
    try:
        logger.info(f"Connecting to IB TWS/Gateway at {IB_HOST}:{IB_PORT} with ClientID {IB_CLIENT_ID}...")
        await ib.connectAsync(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=15)
        logger.info("Successfully connected to IB.")

        # Wait for connection readiness by making an initial request
        # SWITCHING to reqAccountSummaryAsync for a non-blocking check
        logger.info("Requesting account summary snapshot to confirm connection readiness...")
        try:
            # Request a common tag like NetLiquidation
            summary = await ib.reqAccountSummaryAsync(account='All', tags='NetLiquidation')
            if summary:
                logger.info(f"Account summary received (e.g., {summary[0].tag}={summary[0].value}). Connection ready.")
            else:
                logger.warning("Account summary request returned empty, but proceeding. Connection might be okay.")
        except Exception as e:
            logger.error(f"Error during account summary request: {e}. Aborting.", exc_info=True)
            return # Exit if the readiness check itself fails

        # Define the contract
        contract = create_contract(SYMBOL, EXCHANGE, CURRENCY)
        # Qualify the contract (resolves specifics like conId, primaryExchange)
        logger.info(f"Qualifying contract for {SYMBOL}...")
        qual_contracts = await ib.qualifyContractsAsync(contract)
        if not qual_contracts:
            logger.error(f"Could not qualify contract for {SYMBOL}. Exiting.")
            return
        contract = qual_contracts[0] # Use the qualified contract
        logger.info(f"Qualified contract: {contract}")

        # Initial run
        await run_strategy()

        # Schedule periodic runs
        while True:
            await asyncio.sleep(RUN_INTERVAL_SECONDS)
            if ib.isConnected():
                 await run_strategy()
            else:
                 logger.warning("IB Disconnected. Attempting to reconnect...")
                 try:
                     await ib.connectAsync(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=15)
                     logger.info("Reconnected successfully.")
                 except Exception as e:
                     logger.error(f"Reconnection failed: {e}. Will retry later.")
                     # Optional: Implement exponential backoff here


    except ConnectionRefusedError:
        logger.error(f"Connection refused. Is TWS/Gateway running and API enabled on port {IB_PORT}?")
    except asyncio.TimeoutError:
         logger.error(f"Connection timed out. Check network and TWS/Gateway settings.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}") # Log full traceback
    finally:
        if ib and ib.isConnected():
            logger.info("Disconnecting from IB.")
            ib.disconnect()

if __name__ == "__main__":
    # Use nest_asyncio if running in Jupyter/Spyder/etc.
    # util.patchAsyncio() # Deprecated, use nest_asyncio
    try:
        import nest_asyncio
        nest_asyncio.apply()
        logger.info("Applied nest_asyncio patch.")
    except ImportError:
        logger.info("nest_asyncio not found. Skipping patch.")

    # Run the main async loop
    try:
        # ib_insync uses its own loop management now usually
        # asyncio.run(main())
        # Or using ib_insync's run
        util.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except Exception as e:
         logger.critical(f"Fatal error in main execution: {e}", exc_info=True)