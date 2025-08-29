import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_features_for_visitor(visitor_id: int, events_df: pd.DataFrame) -> pd.DataFrame:
    final_feature_list = [
        'view_count', 
        'addtocart_count',
        'unique_items_viewed', 
        'add_to_cart_rate',
        'recency_days',
        'num_sessions',
        'avg_events_per_session'
    ]

    user_events = events_df[events_df['visitorid'] == visitor_id].copy()

    if user_events.empty:
        return pd.DataFrame(columns=final_feature_list)

    user_events['timestamp_dt'] = pd.to_datetime(user_events['timestamp'], unit='ms')
    user_events['timestamp_dt'] = user_events['timestamp_dt'].dt.tz_localize('UTC')
    
    view_count = user_events[user_events['event'] == 'view'].shape[0]
    addtocart_count = user_events[user_events['event'] == 'addtocart'].shape[0]
    unique_items_viewed = user_events[user_events['event'] == 'view']['itemid'].nunique()
    
    add_to_cart_rate = addtocart_count / view_count if view_count > 0 else 0
    
    last_event_time = user_events['timestamp_dt'].max()
    # Compute dataset-wide latest timestamp (analysis cutoff)
    if 'timestamp_dt' in events_df.columns:
        dataset_ts_series = pd.to_datetime(events_df['timestamp_dt'], errors='coerce')
    else:
        dataset_ts_series = pd.to_datetime(events_df['timestamp'], unit='ms', errors='coerce')
    dataset_last_time = dataset_ts_series.max()
    if dataset_last_time.tzinfo is None:
        dataset_last_time = dataset_last_time.tz_localize('UTC')

    # Recency is days between user's last activity and dataset's latest date
    recency_days = (dataset_last_time - last_event_time).days
    
    user_events = user_events.sort_values(by='timestamp_dt')
    time_diffs = user_events['timestamp_dt'].diff()
    session_timeout = pd.Timedelta(minutes=30)
    is_new_session = (time_diffs > session_timeout) | (time_diffs.isnull())
    user_events['session_id'] = is_new_session.cumsum()
    
    num_sessions = user_events['session_id'].nunique()
    avg_events_per_session = user_events.groupby('session_id')['event'].count().mean()

    features_dict = {
        'view_count': [view_count],
        'addtocart_count': [addtocart_count],
        'unique_items_viewed': [unique_items_viewed],
        'add_to_cart_rate': [add_to_cart_rate],
        'recency_days': [recency_days],
        'num_sessions': [num_sessions],
        'avg_events_per_session': [avg_events_per_session]
    }
    
    return pd.DataFrame(features_dict)