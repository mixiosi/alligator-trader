# -*- coding: utf-8 -*-
import asyncio
import logging
from datetime import datetime, timedelta
import time # Keep time for potential sleeps if needed

import pandas as pd
import pandas_ta as ta
from ib_insync import IB, Stock, Contract, MarketOrder, LimitOrder, StopOrder, Order, Trade, util # Added Order, Trade

# --- Configuration ---
# TWS/Gateway Connection
IB_HOST = '127.0.0.1'
IB_PORT = 7497  # 7497 for TWS Paper, 7496 for TWS Live, 4002 for Gateway Paper, 4001 for Gateway Live
IB_CLIENT_ID = 2 # Use a different client ID than bb_ml_stable3 if running concurrently

# Trading Parameters
SYMBOL = 'MSTR'
EXCHANGE = 'SMART'
CURRENCY = 'USD'
PRIMARY_TIMEFRAME = '4 hours' # Timeframe for primary signals (CMO, StochRSI, current Alligator, ATR)
LONGTERM_TIMEFRAME = '6 hours' # Timeframe for long-term confirmation (ADX, LT Alligator, LT EMA)
ORDER_SIZE_PERCENT = 0.01 # Risk 1% of portfolio value per trade (adjust as needed)
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
RUN_INTERVAL_SECONDS = 4 * 60 * 60 # Check every 4 hours (adjust as needed)
# RUN_INTERVAL_SECONDS = 60 * 5 # For testing: Check every 5 minutes
MAX_DATA_REQUEST_BARS = 300 # How many bars to fetch (ensure enough for longest lookback)
LOG_LEVEL = logging.INFO # DEBUG, INFO, WARNING, ERROR
ACCOUNT_SYNC_WAIT_SECONDS = 5 # How long to wait after connection for account data

# --- Logging Setup ---
# Use ib_insync's logging integration
# util.logToConsole(LOG_LEVEL) # Uncomment for detailed ib_insync logs
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use __name__ for clarity

# --- Global Variables ---
ib: IB = None # IB connection object, type hint added
contract: Contract = None # Stock contract object
active_trade_pending: bool = False # Flag to prevent overlapping orders
active_parent_order_id: int = None # Store the ID of the pending parent order

# --- Helper Functions ---

def create_contract(symbol, exchange, currency):
    """Creates an IB stock contract object."""
    return Stock(symbol, exchange, currency)

# Function from working example, adapted slightly
def round_price(price, tick_size=0.01):
    """Rounds price to the nearest tick size."""
    return round(round(price / tick_size) * tick_size, 2) # Ensure 2 decimal places common for USD stocks


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
        # Keep original column names from IB for clarity
        return df

    except ConnectionError as ce:
        logger.error(f"ConnectionError fetching historical data for {contract.symbol} - {timeframe}: {ce}. Check TWS/Gateway connection.")
        # Potentially trigger a reconnection attempt or shutdown
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching historical data for {contract.symbol} - {timeframe}: {e}", exc_info=True)
        return pd.DataFrame()

def calculate_indicators(df: pd.DataFrame, timeframe_label: str):
    """Calculates and adds technical indicators to the DataFrame."""
    if df.empty or len(df) < max(ADX_PERIOD, CMO_PERIOD, STOCHRSI_PERIOD, ALLIGATOR_JAW, EMA_LONGTERM_PERIOD, ATR_PERIOD):
        logger.warning(f"Cannot calculate indicators on insufficient data ({len(df)} bars) for {timeframe_label}")
        return pd.DataFrame() # Return empty if not enough data

    logger.debug(f"Calculating indicators for {timeframe_label} timeframe")
    df_out = df.copy() # Work on a copy to avoid SettingWithCopyWarning
    try:
        # Alligator
        df_out.ta.alligator(jaw_length=ALLIGATOR_JAW, teeth_length=ALLIGATOR_TEETH, lips_length=ALLIGATOR_LIPS, append=True)
        df_out.rename(columns={
            f'ALLIGATORj_{ALLIGATOR_JAW}_{ALLIGATOR_TEETH}_{ALLIGATOR_LIPS}': f'alligator_jaw_{timeframe_label}',
            f'ALLIGATORt_{ALLIGATOR_JAW}_{ALLIGATOR_TEETH}_{ALLIGATOR_LIPS}': f'alligator_teeth_{timeframe_label}',
            f'ALLIGATORl_{ALLIGATOR_JAW}_{ALLIGATOR_TEETH}_{ALLIGATOR_LIPS}': f'alligator_lips_{timeframe_label}'
        }, inplace=True)

        # ADX (only needed for long-term timeframe)
        if timeframe_label == 'longterm':
            df_out.ta.adx(length=ADX_PERIOD, append=True)
            df_out.rename(columns={f'ADX_{ADX_PERIOD}': 'adx'}, inplace=True)

        # CMO (only needed for primary timeframe)
        if timeframe_label == 'primary':
            df_out.ta.cmo(length=CMO_PERIOD, append=True)
            df_out.rename(columns={f'CMO_{CMO_PERIOD}': 'cmo'}, inplace=True)

        # Stochastic RSI (only needed for primary timeframe)
        if timeframe_label == 'primary':
            stoch_rsi = df_out.ta.stochrsi(length=STOCHRSI_PERIOD, rsi_length=STOCHRSI_PERIOD, k=STOCHRSI_K, d=STOCHRSI_D, append=True)
            # Check if columns were actually added (can fail if input data is constant)
            if f'STOCHRSIk_{STOCHRSI_PERIOD}_{STOCHRSI_PERIOD}_{STOCHRSI_K}_{STOCHRSI_D}' in df_out.columns:
                df_out.rename(columns={
                    f'STOCHRSIk_{STOCHRSI_PERIOD}_{STOCHRSI_PERIOD}_{STOCHRSI_K}_{STOCHRSI_D}': 'stochrsi_k',
                    f'STOCHRSId_{STOCHRSI_PERIOD}_{STOCHRSI_PERIOD}_{STOCHRSI_K}_{STOCHRSI_D}': 'stochrsi_d'
                }, inplace=True)
            else:
                logger.warning(f"StochRSI calculation did not add columns for {timeframe_label}. Adding NaNs.")
                df_out['stochrsi_k'] = pd.NA
                df_out['stochrsi_d'] = pd.NA


        # EMA (only needed for long-term timeframe)
        if timeframe_label == 'longterm':
             df_out.ta.ema(length=EMA_LONGTERM_PERIOD, append=True)
             df_out.rename(columns={f'EMA_{EMA_LONGTERM_PERIOD}': f'ema_{EMA_LONGTERM_PERIOD}'}, inplace=True)

        # ATR (only needed for primary timeframe, for SL/TP)
        if timeframe_label == 'primary':
             df_out.ta.atr(length=ATR_PERIOD, append=True)
             df_out.rename(columns={f'ATRr_{ATR_PERIOD}': 'atr'}, inplace=True)

        # Drop initial rows with NaNs from indicator calculations BEFORE returning
        # Keep original index for alignment if needed later
        df_out.dropna(inplace=True)
        logger.debug(f"Finished calculating indicators for {timeframe_label}, {len(df_out)} rows remain after dropna.")

    except Exception as e:
        logger.error(f"Error calculating indicators for {timeframe_label}: {e}", exc_info=True)
        return pd.DataFrame() # Return empty on error
    return df_out


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

# --- Use cached account values ---
def get_account_value(ib_conn: IB, currency: str = 'USD', account_id: str = None):
    """Retrieves the NetLiquidation value from cached account values."""
    try:
        logger.debug("Attempting to retrieve cached account values...")
        account_values = ib_conn.accountValues() # Access cached data
        if not account_values:
            logger.error("Account values cache is empty. Ensure reqAccountUpdates stream is active and data has arrived.")
            return None

        logger.debug(f"Found {len(account_values)} cached items. Searching for NetLiquidation/{currency}...")

        net_liq_item = None
        for item in account_values:
            if item.tag == 'NetLiquidation' and item.currency == currency:
                 # If an account_id is specified, check that too (optional)
                 if account_id is None or item.account == account_id:
                     net_liq_item = item
                     logger.info(f"Found Cached Account Net Liquidation ({currency}, Account: {item.account}): {item.value}")
                     break

        if net_liq_item:
            try:
                return float(net_liq_item.value)
            except (ValueError, TypeError):
                logger.error(f"Could not convert cached account value '{net_liq_item.value}' for {net_liq_item.account} to float.")
                return None
        else:
            logger.warning(f"NetLiquidation ({currency}) not found in cached values" + (f" for account {account_id}." if account_id else "."))
            # Log available tags for debugging
            # available_items = {(item.account, item.tag, item.currency) for item in account_values}
            # logger.debug(f"Available cached items (Account, Tag, Currency): {available_items}")
            return None

    except Exception as e:
         logger.error(f"Error retrieving/processing cached account value: {e}", exc_info=True)
         return None

async def check_existing_position(ib_conn: IB, contract: Contract):
    """Checks if there is an existing position for the contract using cached data."""
    # Use ib_conn.positions() for cached data if reqPositions() was called
    # Or use await ib_conn.reqPositionsAsync() for a live update (slower)
    try:
        # Let's use the cached version assuming reqPositions() runs periodically or at start
        # Note: Might need to call ib.reqPositions() in main loop or periodically if cache isn't updated
        positions = ib_conn.positions() # Access cached positions
        conId_to_check = contract.conId
        if not conId_to_check:
            logger.warning(f"Contract for {contract.symbol} not qualified yet, cannot check position by conId.")
            # Fallback to symbol check? Risky if multiple contracts for same symbol
            return 0

        for pos in positions:
            if pos.contract.conId == conId_to_check:
                logger.info(f"Existing position found for {contract.symbol} (conId {conId_to_check}): {pos.position} shares @ {pos.avgCost}.")
                return pos.position # Returns size (positive for long, negative for short)

        logger.info(f"No existing position found for {contract.symbol} (conId {conId_to_check}) in cache.")
        return 0 # No position

    except Exception as e:
        logger.error(f"Error checking existing position for {contract.symbol}: {e}", exc_info=True)
        return 0 # Assume no position on error

# --- Modified Order Placement ---
async def place_bracket_order(ib_conn: IB, contract: Contract, action: str, quantity: float, entry_price: float, stop_loss_price: float, take_profit_price: float):
    """
    Places a bracket order (Market entry with attached SL and TP).
    Uses explicit Parent/Child orders with transmit flags.
    Returns the parent Trade object if successful, None otherwise.
    """
    global active_parent_order_id # We need to store the parent ID globally

    if action not in ['BUY', 'SELL']:
        logger.error(f"Invalid action for order: {action}")
        return None

    # Ensure contract is qualified
    if not contract.conId:
        logger.error(f"Contract {contract.symbol} is not qualified. Cannot place order.")
        return None

    # Round prices using helper function
    # Entry price for Market order isn't set, but used for logging/reference
    sl_price = round_price(stop_loss_price)
    tp_price = round_price(take_profit_price)
    ref_entry_price = round_price(entry_price) # For logging

    logger.info(f"Preparing {action} bracket order for {quantity} {contract.symbol} @ ~{ref_entry_price:.2f} [SL={sl_price:.2f}, TP={tp_price:.2f}]")

    # --- Create Orders (mimicking bb_ml_stable3 structure) ---
    # 1. Parent Order (Market Entry)
    parent = MarketOrder(action, quantity)
    parent.transmit = False # IMPORTANT: Do not transmit parent yet
    # Optional: Add order reference,tif etc if needed
    # parent.orderRef = f"Alligator_{action}_{int(time.time())}"
    # parent.tif = "GTC" # Or DAY

    # 2. Take Profit Order (Limit Order)
    tp_action = 'SELL' if action == 'BUY' else 'BUY'
    take_profit = LimitOrder(tp_action, quantity, tp_price)
    take_profit.parentId = 0 # Will be set after parent is placed
    take_profit.transmit = False # IMPORTANT: Do not transmit TP yet
    # take_profit.tif = "GTC"

    # 3. Stop Loss Order (Stop Order)
    sl_action = 'SELL' if action == 'BUY' else 'BUY'
    stop_loss = StopOrder(sl_action, quantity, sl_price)
    stop_loss.parentId = 0 # Will be set after parent is placed
    stop_loss.transmit = True # IMPORTANT: Transmit the LAST order in the bracket
    # stop_loss.tif = "GTC"

    # --- Place Orders ---
    try:
        # Place Parent Order
        parent_trade = ib_conn.placeOrder(contract, parent)
        logger.info(f"Placing Parent {parent.action} Order {parent_trade.order.orderId} for {contract.symbol}...")

        # Wait briefly for the orderId to be assigned and status potentially updated
        await parent_trade.orderStatusEvent # Wait for initial status event

        if not parent_trade.order.orderId or parent_trade.order.permId == 0:
             logger.error(f"Failed to get valid OrderId/PermId for parent order ({parent_trade.orderStatus.status}). Cancelling attempt.")
             # Attempt to cancel if possible (may not have been transmitted)
             if parent_trade.order.orderId:
                 ib_conn.cancelOrder(parent_trade.order)
             return None

        # Successfully got parent OrderId, store it and update children
        parent_order_id = parent_trade.order.orderId
        active_parent_order_id = parent_order_id # Store globally for status tracking
        logger.info(f"Parent Order {parent_order_id} received by TWS ({parent_trade.orderStatus.status}). Placing child orders...")

        take_profit.parentId = parent_order_id
        stop_loss.parentId = parent_order_id

        # Place Take Profit Order
        tp_trade = ib_conn.placeOrder(contract, take_profit)
        logger.info(f"Placing TP Order {tp_trade.order.orderId} (Parent: {parent_order_id})...")

        # Place Stop Loss Order (this one transmits all)
        sl_trade = ib_conn.placeOrder(contract, stop_loss)
        logger.info(f"Placing SL Order {sl_trade.order.orderId} (Parent: {parent_order_id}) - Transmitting group...")

        # Optional: Wait briefly for child order statuses if needed, but rely on onOrderStatus handler
        # await asyncio.sleep(1)

        logger.info(f"Bracket order group (Parent: {parent_order_id}) submitted for {contract.symbol}.")
        return parent_trade # Return the parent trade object for reference

    except Exception as e:
        logger.error(f"Error placing {action} bracket order for {contract.symbol}: {e}", exc_info=True)
        # Attempt to cancel any orders that might have been placed partially
        global active_trade_pending
        if active_parent_order_id: # If parent was placed
             logger.warning(f"Attempting to cancel orders associated with Parent ID {active_parent_order_id} due to error.")
             # This might not work perfectly if TWS already processed parts
             ib_conn.reqGlobalCancel() # More drastic, cancels all orders
             # Or try cancelling specific parent ID if possible (less reliable for bracket)
             # ib_conn.cancelOrder(ib.order(orderId=active_parent_order_id)) # Needs order object?

        active_trade_pending = False # Reset pending flag on failure
        active_parent_order_id = None
        return None

# --- Event Handler for Order Status ---
def onOrderStatus(trade: Trade):
    """Callback for order status updates."""
    global active_trade_pending, active_parent_order_id

    # Log every status update for debugging
    # logger.debug(f"Order Status Update: OrderId={trade.order.orderId}, ParentId={trade.order.parentId}, Status={trade.orderStatus.status}, Filled={trade.orderStatus.filled}, Symbol={trade.contract.symbol}")

    # Check if this update pertains to the parent order we are actively tracking
    if active_parent_order_id is not None and trade.order.orderId == active_parent_order_id:
        logger.info(f"Status update for ACTIVE PARENT order {active_parent_order_id}: {trade.orderStatus.status}")
        if trade.isDone(): # isDone() checks for Filled, Cancelled, Expired, etc.
            logger.info(f"** Parent order {active_parent_order_id} ({trade.contract.symbol}) has reached a final state ({trade.orderStatus.status}). Clearing pending flag. **")
            active_trade_pending = False
            active_parent_order_id = None # Reset the tracked ID
        # Optional: Could clear the flag earlier, e.g., on 'Submitted', if desired, but 'isDone' is safer
        # elif trade.orderStatus.status == 'Submitted':
        #     logger.info(f"Parent order {active_parent_order_id} is Submitted. Still pending execution.")

    # Optional: Log status of child orders if needed for detailed tracking
    # elif active_parent_order_id is not None and trade.order.parentId == active_parent_order_id:
    #    logger.debug(f"Status update for CHILD order {trade.order.orderId} (Parent: {active_parent_order_id}): {trade.orderStatus.status}")

# --- Main Strategy Logic ---
async def run_strategy():
    """Main loop to fetch data, check strategy conditions, and place orders."""
    global ib, contract, active_trade_pending # Make flag global

    if not ib or not ib.isConnected():
        logger.error("IB connection is not available.")
        return

    if not contract or not contract.conId: # Ensure contract is qualified
        logger.error("Contract is not defined or not qualified.")
        # Attempt requalification?
        try:
             logger.warning("Attempting to re-qualify contract...")
             qual_contracts = await ib.qualifyContractsAsync(create_contract(SYMBOL, EXCHANGE, CURRENCY))
             if qual_contracts:
                 contract = qual_contracts[0]
                 logger.info(f"Re-qualified contract: {contract}")
             else:
                 logger.error("Failed to re-qualify contract.")
                 return
        except Exception as e:
             logger.error(f"Error re-qualifying contract: {e}")
             return

    logger.info(f"--- Running Strategy Check for {SYMBOL} at {datetime.now()} ---")

    # --- 1. Check if Trade Pending ---
    if active_trade_pending:
        logger.info(f"Skipping strategy check: Active trade (Parent ID: {active_parent_order_id}) is pending confirmation/completion.")
        return

    # --- 2. Check Existing Position ---
    # Use cached position data, assuming it's updated via reqPositions() or stream
    current_position = check_existing_position(ib, contract) # Changed to non-async version using cache
    if current_position != 0:
        logger.info(f"Already have a position of {current_position} for {SYMBOL}. Skipping new entry check.")
        return # Simple strategy: only enter if flat

    # --- 3. Get Account Value ---
    account_value = get_account_value(ib, CURRENCY) # Uses cached value
    if account_value is None:
        logger.error("Could not get account value from cache. Ensure account updates are running. Skipping trade.")
        return

    # --- 4. Get Historical Data ---
    # Duration string calculation (simplified)
    duration_str = f"{MAX_DATA_REQUEST_BARS * 2} D" # Fetch ample data

    df_primary = await get_historical_data(ib, contract, PRIMARY_TIMEFRAME, duration_str)
    if df_primary.empty:
        logger.warning(f"Could not get primary timeframe ({PRIMARY_TIMEFRAME}) data. Skipping.")
        return

    df_longterm = await get_historical_data(ib, contract, LONGTERM_TIMEFRAME, duration_str)
    if df_longterm.empty:
        logger.warning(f"Could not get long-term timeframe ({LONGTERM_TIMEFRAME}) data. Skipping.")
        return

    # --- 5. Calculate Indicators ---
    # Pass copies to avoid modifying original dfs if needed elsewhere
    df_primary_ind = calculate_indicators(df_primary.copy(), 'primary')
    df_longterm_ind = calculate_indicators(df_longterm.copy(), 'longterm')

    if df_primary_ind.empty or df_longterm_ind.empty:
        logger.warning("Indicator calculation resulted in empty data or insufficient rows. Skipping check.")
        return

    # --- 6. Get Latest Data Point ---
    try:
        # Use .iloc[-1] to get the last row
        latest_primary = df_primary_ind.iloc[-1]
        # Find corresponding longterm bar (can be slightly tricky with different timeframes)
        # Simplest: use the latest longterm bar available
        latest_longterm = df_longterm_ind.iloc[-1]
        current_price = latest_primary['close'] # Use primary close price for checks & orders

        # Check alignment (optional but good practice)
        time_diff = abs(latest_primary.name - latest_longterm.name)
        if time_diff > pd.Timedelta(hours=max(int(PRIMARY_TIMEFRAME.split()[0]), int(LONGTERM_TIMEFRAME.split()[0])) * 1.5): # Allow some lag
             logger.warning(f"Significant time difference between latest primary ({latest_primary.name}) and longterm ({latest_longterm.name}) bars: {time_diff}. Check data fetching.")
             # Decide whether to proceed or skip if alignment is critical

    except IndexError:
        logger.warning("Not enough data rows after indicator calculation to get latest data. Skipping check.")
        return
    except Exception as e:
        logger.error(f"Error accessing latest data points: {e}", exc_info=True)
        return


    # Check for NaN values in latest row again (should have been dropped, but good sanity check)
    if latest_primary.isnull().any() or latest_longterm.isnull().any():
         logger.warning(f"Latest data row contains NaN values after processing. Skipping check. Primary NaNs: {latest_primary.isnull().sum()}, Longterm NaNs: {latest_longterm.isnull().sum()}")
         return

    # --- 7. Evaluate Strategy Conditions ---
    # Ensure all required columns exist before accessing them
    required_primary_cols = ['close', 'atr', 'cmo', 'stochrsi_k', 'stochrsi_d', 'alligator_jaw_primary', 'alligator_teeth_primary', 'alligator_lips_primary']
    required_longterm_cols = ['close', 'adx', f'ema_{EMA_LONGTERM_PERIOD}', 'alligator_jaw_longterm', 'alligator_teeth_longterm', 'alligator_lips_longterm']

    if not all(col in latest_primary.index for col in required_primary_cols):
         missing_cols = [col for col in required_primary_cols if col not in latest_primary.index]
         logger.error(f"Missing required columns in latest primary data: {missing_cols}")
         return
    if not all(col in latest_longterm.index for col in required_longterm_cols):
         missing_cols = [col for col in required_longterm_cols if col not in latest_longterm.index]
         logger.error(f"Missing required columns in latest long-term data: {missing_cols}")
         return


    primary_trend = determine_alligator_trend(latest_primary['alligator_jaw_primary'], latest_primary['alligator_teeth_primary'], latest_primary['alligator_lips_primary'])
    longterm_trend = determine_alligator_trend(latest_longterm['alligator_jaw_longterm'], latest_longterm['alligator_teeth_longterm'], latest_longterm['alligator_lips_longterm'])
    longterm_ema_trend_up = current_price > latest_longterm[f'ema_{EMA_LONGTERM_PERIOD}']

    # Log indicator values
    logger.info(f"Data Time: Primary={latest_primary.name}, LongTerm={latest_longterm.name}")
    logger.info(f"Indicators: Price={current_price:.2f}, P_Trend={primary_trend}, L_Trend={longterm_trend}, ADX={latest_longterm['adx']:.2f}, P_CMO={latest_primary['cmo']:.2f}, P_StochK={latest_primary['stochrsi_k']:.2f}, Above_L_EMA={longterm_ema_trend_up}, P_ATR={latest_primary['atr']:.3f}")


    trade_action = None

    # --- Long Entry Conditions ---
    if (primary_trend == "UPTREND" and
        latest_longterm['adx'] > ADX_THRESHOLD and
        longterm_trend == "UPTREND" and
        longterm_ema_trend_up and
        latest_primary['cmo'] > CMO_LONG_THRESHOLD and
        latest_primary['stochrsi_k'] < STOCHRSI_OVERSOLD):
        logger.info(">>> LONG ENTRY CONDITIONS MET <<<")
        trade_action = 'BUY'

    # --- Short Entry Conditions ---
    elif (primary_trend == "DOWNTREND" and
          latest_longterm['adx'] > ADX_THRESHOLD and
          longterm_trend == "DOWNTREND" and
          not longterm_ema_trend_up and
          latest_primary['cmo'] < CMO_SHORT_THRESHOLD and
          latest_primary['stochrsi_k'] > STOCHRSI_OVERBOUGHT):
        logger.info(">>> SHORT ENTRY CONDITIONS MET <<<")
        trade_action = 'SELL'

    else:
        logger.info("Conditions not met for entry.")

    # --- 8. Place Order If Conditions Met ---
    if trade_action:
        atr_value = latest_primary['atr']
        if pd.isna(atr_value) or atr_value <= 0: # Check ATR validity
            logger.error(f"ATR value ({atr_value}) is invalid. Cannot calculate SL/TP or position size.")
            return

        # Calculate Stop Loss and Take Profit
        if trade_action == 'BUY':
            stop_loss_val = current_price - (atr_value * SL_ATR_MULTIPLIER)
            take_profit_val = current_price + (atr_value * TP_ATR_MULTIPLIER)
            risk_per_share = current_price - stop_loss_val
        else: # SELL
            stop_loss_val = current_price + (atr_value * SL_ATR_MULTIPLIER)
            take_profit_val = current_price - (atr_value * TP_ATR_MULTIPLIER)
            risk_per_share = stop_loss_val - current_price

        if risk_per_share <= 0:
            logger.error(f"Risk per share is zero or negative ({risk_per_share:.2f}). Cannot calculate position size. Check ATR and multipliers.")
            return

        # Calculate Position Size (using logic similar to bb_ml_stable3)
        max_risk_amount = account_value * ORDER_SIZE_PERCENT
        quantity = int(max_risk_amount / risk_per_share)
        quantity = max(1, min(quantity, 10000)) # Ensure min 1 share, add reasonable max cap

        logger.info(f"Position Size Calc: AccountValue={account_value:.2f}, RiskPercent={ORDER_SIZE_PERCENT:.2%}, MaxRiskAmt={max_risk_amount:.2f}, RiskPerShare={risk_per_share:.2f}, InitialQty={int(max_risk_amount / risk_per_share)}, FinalQty={quantity}")


        if quantity <= 0:
            logger.warning(f"Calculated order quantity is {quantity}. Check risk % or account value/ATR.")
            return

        # Set the pending flag BEFORE placing the order
        logger.info("Setting active_trade_pending = True before placing order.")
        active_trade_pending = True

        # Place the bracket order
        parent_trade_obj = await place_bracket_order(
            ib, contract, trade_action, quantity,
            current_price, stop_loss_val, take_profit_val
        )

        if parent_trade_obj is None:
            logger.error("Order placement failed. active_trade_pending flag should have been reset by place_bracket_order.")
            # Double-check reset just in case
            active_trade_pending = False
            active_parent_order_id = None
        else:
            logger.info(f"Order placement initiated successfully for Parent ID {parent_trade_obj.order.orderId}. Waiting for onOrderStatus callback to clear pending flag.")
            # The onOrderStatus handler will set active_trade_pending=False when appropriate


# --- Main Execution ---
async def main():
    global ib, contract
    ib = IB()

    # Register event handlers BEFORE connecting
    ib.orderStatusEvent += onOrderStatus
    # ib.errorEvent += onError # Example: Define an onError handler if needed
    # ib.disconnectedEvent += onDisconnected # Example: Handle disconnections

    try:
        logger.info(f"Connecting to IB TWS/Gateway at {IB_HOST}:{IB_PORT} with ClientID {IB_CLIENT_ID}...")
        await ib.connectAsync(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=20) # Increased timeout
        logger.info(f"Successfully connected to IB. Server version: {ib.serverVersion()}. TWS Time: {ib.reqCurrentTime()}")

        # Start account updates stream
        logger.info("Starting account updates subscription...")
        try:
            await ib.reqAccountUpdatesAsync(account='All') # Requires 'account' arg
            logger.info(f"Account updates subscription initiated. Waiting {ACCOUNT_SYNC_WAIT_SECONDS}s for initial data...")
            await asyncio.sleep(ACCOUNT_SYNC_WAIT_SECONDS)
            # Additionally request positions to populate cache initially
            await ib.reqPositionsAsync()
            logger.info("Initial account data sync period complete.")
        except Exception as e:
            logger.error(f"Error initiating account updates/positions: {e}. Aborting.", exc_info=True)
            return

        # Define and qualify the contract
        contract = create_contract(SYMBOL, EXCHANGE, CURRENCY)
        logger.info(f"Qualifying contract for {SYMBOL}...")
        qual_contracts = await ib.qualifyContractsAsync(contract)
        if not qual_contracts:
            logger.error(f"Could not qualify contract for {SYMBOL}. Exiting.")
            return
        contract = qual_contracts[0]
        logger.info(f"Qualified contract: {contract}")


        # Initial run
        await run_strategy()

        # Schedule periodic runs
        while True:
            await asyncio.sleep(RUN_INTERVAL_SECONDS)
            # Check connection before running strategy
            if not ib.isConnected():
                 logger.warning("IB Disconnected. Stopping strategy loop.")
                 break # Exit the loop if disconnected

            # Optional: Refresh positions cache periodically if needed
            # logger.debug("Requesting positions update...")
            # await ib.reqPositionsAsync()
            # await asyncio.sleep(1) # Small delay after request

            await run_strategy()


    except ConnectionRefusedError:
        logger.error(f"Connection refused. Is TWS/Gateway running and API enabled on port {IB_PORT}?")
    except asyncio.TimeoutError:
         logger.error(f"Connection timed out. Check network and TWS/Gateway settings.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred in main loop: {e}")
    finally:
        if ib and ib.isConnected():
            logger.info("Disconnecting from IB.")
            # Cancel pending order status tracking? No explicit cancellation needed.
            ib.disconnect()
        logger.info("Program finished.")


if __name__ == "__main__":
    # Use nest_asyncio if running in Jupyter/Spyder/etc.
    try:
        import nest_asyncio
        nest_asyncio.apply()
        logger.info("Applied nest_asyncio patch.")
    except ImportError:
        logger.info("nest_asyncio not found. Skipping patch.")

    # Run the main async loop using ib_insync's utility
    try:
        util.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user (KeyboardInterrupt).")
    except SystemExit:
        logger.info("Program exited.") # Catch potential sys.exit() calls
    except Exception as e:
         logger.critical(f"Fatal error during main execution: {e}", exc_info=True)