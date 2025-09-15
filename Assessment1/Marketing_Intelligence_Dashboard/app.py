# app.py - cleaned for readability (functionality preserved)
# - Reformatted, added docstrings and clearer comments
# - No logic changes, only cosmetic improvements to help maintenance

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from typing import Optional

st.set_page_config(layout="wide", page_title="Marketing Intelligence Dashboard")

# ----------------------
# Helper functions
# ----------------------

def load_csv_safe(path: str, parse_dates: list = ['date']) -> Optional[pd.DataFrame]:
    """Safely load a CSV file using pandas.

    Returns None and shows a Streamlit error if the file cannot be read.
    """
    try:
        df = pd.read_csv(path, parse_dates=parse_dates, dayfirst=False, keep_default_na=True)
    except FileNotFoundError:
        st.error(f"File not found: {path}")
        return None
    except Exception as e:
        st.error(f"Error reading {path}: {e}")
        return None
    return df


def standardize_marketing_df(df: pd.DataFrame, channel_name: str) -> pd.DataFrame:
    """Normalize column names and types for marketing channel DataFrames.

    Ensures columns: date, tactic, state, campaign, impressions, clicks, spend, attributed_revenue
    are present and coerced to appropriate types. Rows with invalid dates are dropped.
    """
    # normalize column names to lowercase and trim spaces
    df = df.rename(columns=lambda x: x.strip().lower())

    # map likely variants to consistent names
    col_map = {}
    for c in df.columns:
        key = c.replace(' ', '')
        if key in ('attributedrevenue', 'attributed_revenue', 'attributedrevenue'):
            col_map[c] = 'attributed_revenue'
        if key in ('impressions', 'impression'):
            col_map[c] = 'impressions'
        if key == 'clicks':
            col_map[c] = 'clicks'
        if key in ('spend', 'cost'):
            col_map[c] = 'spend'
        if key == 'campaign':
            col_map[c] = 'campaign'
        if key == 'tactic':
            col_map[c] = 'tactic'
        if key == 'state':
            col_map[c] = 'state'

    df = df.rename(columns=col_map)

    # ensure date column is datetime and drop invalid dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # coerce numeric columns to numbers; missing columns become 0
    for numcol in ['impressions', 'clicks', 'spend', 'attributed_revenue']:
        if numcol in df.columns:
            df[numcol] = pd.to_numeric(df[numcol], errors='coerce').fillna(0)
        else:
            df[numcol] = 0

    # set channel name so we can differentiate sources after concatenation
    df['channel'] = channel_name

    # drop rows without a valid date
    df = df[~df['date'].isna()].copy()
    return df


def standardize_business_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and types for the business DataFrame.

    Expected (or interpreted) columns:
      - date, orders, new_orders, new_customers, revenue, gross_profit, cogs
    """
    df = df.rename(columns=lambda x: x.strip().lower())

    # normalize common variants into target column names
    col_map = {}
    for c in df.columns:
        key = c.lower().replace('#', '').replace(' ', '').replace('_', '')
        if 'order' in key and 'new' not in key:
            col_map[c] = 'orders'
        if 'neworders' in key or ('new' in key and 'orders' in key):
            col_map[c] = 'new_orders'
        if 'newcustomers' in key or 'newcustomer' in key:
            col_map[c] = 'new_customers'
        if 'total' in key and 'revenue' in key:
            col_map[c] = 'revenue'
        if 'gross' in key and 'profit' in key:
            col_map[c] = 'gross_profit'
        if 'cogs' in key or 'costofgoods' in key:
            col_map[c] = 'cogs'

    df = df.rename(columns=col_map)

    # ensure date column is datetime and drop invalid dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # coerce expected numeric columns to numeric; missing become 0
    for n in ['orders', 'new_orders', 'new_customers', 'revenue', 'gross_profit', 'cogs']:
        if n in df.columns:
            df[n] = pd.to_numeric(df[n], errors='coerce').fillna(0)
        else:
            df[n] = 0

    df = df[~df['date'].isna()].copy()
    return df


# ----------------------
# Load files
# ----------------------
st.title("Marketing Intelligence Dashboard")
st.markdown("Load CSVs: `facebook.csv`, `google.csv`, `tiktok.csv`, `business.csv` (place these files in the same folder).")

# fb = load_csv_safe("facebook.csv")
# gg = load_csv_safe("google.csv")
# tt = load_csv_safe("tiktok.csv")
# biz = load_csv_safe("business.csv")

# # If any file couldn't be loaded, stop and show a warning
# if fb is None or gg is None or tt is None or biz is None:
#     st.warning("One or more CSVs are missing. Please ensure facebook.csv, google.csv, tiktok.csv, and business.csv exist.")
#     st.stop()

facebook_file = st.file_uploader("Upload facebook.csv", type=["csv"])
google_file = st.file_uploader("Upload google.csv", type=["csv"])
tiktok_file = st.file_uploader("Upload tiktok.csv", type=["csv"])
business_file = st.file_uploader("Upload business.csv", type=["csv"])

if facebook_file and google_file and tiktok_file and business_file:
    fb = pd.read_csv(facebook_file)
    gg = pd.read_csv(google_file)
    tt = pd.read_csv(tiktok_file)
    biz = pd.read_csv(business_file)
    st.success("All CSVs loaded successfully!")

    # ----------------------
    # Clean & prepare data
    # ----------------------
    fb = standardize_marketing_df(fb, "Facebook")
    gg = standardize_marketing_df(gg, "Google")
    tt = standardize_marketing_df(tt, "TikTok")

    # combine marketing channels
    marketing = pd.concat([fb, gg, tt], ignore_index=True)

    # ensure expected columns exist on the combined marketing df
    for c in ['tactic', 'state', 'campaign', 'impressions', 'clicks', 'spend', 'attributed_revenue', 'channel', 'date']:
        if c not in marketing.columns:
            # numeric fields -> 0, others -> empty string
            marketing[c] = 0 if c in ['impressions', 'clicks', 'spend', 'attributed_revenue'] else ""

    # standardize business df
    biz = standardize_business_df(biz)

    # aggregate marketing by date + channel (daily)
    marketing_daily_channel = (
        marketing
        .groupby(['date', 'channel'], as_index=False)
        .agg({'impressions': 'sum', 'clicks': 'sum', 'spend': 'sum', 'attributed_revenue': 'sum'})
    )

    # campaign-level aggregation (useful for top campaigns)
    marketing_campaign = (
        marketing
        .groupby(['date', 'channel', 'campaign'], as_index=False)
        .agg({'impressions': 'sum', 'clicks': 'sum', 'spend': 'sum', 'attributed_revenue': 'sum'})
    )

    # totals by date across channels
    marketing_total_by_date = (
        marketing_daily_channel
        .groupby('date', as_index=False)
        .agg({'impressions': 'sum', 'clicks': 'sum', 'spend': 'sum', 'attributed_revenue': 'sum'})
        .rename(columns=lambda x: f"marketing_{x}" if x != 'date' else x)
    )

    # merge business + marketing totals
    df = pd.merge(biz, marketing_total_by_date, on='date', how='left')
    # fill NaNs for marketing columns
    for col in ['marketing_impressions', 'marketing_clicks', 'marketing_spend', 'marketing_attributed_revenue']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0

    # ----------------------
    # Derived metrics
    # ----------------------
    # CTR, CPC, CPM, ROAS, marketing contribution to revenue
    df['ctr'] = np.where(df['marketing_impressions'] > 0, df['marketing_clicks'] / df['marketing_impressions'], np.nan)
    df['cpc'] = np.where(df['marketing_clicks'] > 0, df['marketing_spend'] / df['marketing_clicks'], np.nan)
    # CPM: cost per thousand impressions
    df['cpm'] = np.where(df['marketing_impressions'] > 0, df['marketing_spend'] / (df['marketing_impressions'] / 1000), np.nan)
    df['roas'] = np.where(df['marketing_spend'] > 0, df['marketing_attributed_revenue'] / df['marketing_spend'], np.nan)
    # marketing contribution as share of total business revenue
    df['marketing_pct_of_revenue'] = np.where(df['revenue'] > 0, df['marketing_attributed_revenue'] / df['revenue'], 0)

    # 7-day moving averages for smoothing
    df = df.sort_values('date').reset_index(drop=True)
    df['spend_7d_ma'] = df['marketing_spend'].rolling(window=7, min_periods=1).mean()
    df['revenue_7d_ma'] = df['revenue'].rolling(window=7, min_periods=1).mean()

    # ----------------------
    # Sidebar filters
    # ----------------------
    st.sidebar.header("Filters")
    min_date = marketing['date'].min()
    max_date = marketing['date'].max()
    if pd.isna(min_date):
        min_date = df['date'].min()
    if pd.isna(max_date):
        max_date = df['date'].max()

    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    channels = sorted(marketing['channel'].dropna().unique().tolist())
    selected_channels = st.sidebar.multiselect("Channel", options=channels, default=channels)

    states = sorted(marketing['state'].astype(str).unique().tolist())
    selected_states = st.sidebar.multiselect("State", options=states, default=[])

    tactics = sorted(marketing['tactic'].astype(str).unique().tolist())
    selected_tactics = st.sidebar.multiselect("Tactic", options=tactics, default=[])

    # Campaign filter optional
    campaigns = sorted(marketing['campaign'].astype(str).unique().tolist())
    selected_campaigns = st.sidebar.multiselect("Campaign (optional)", options=campaigns, default=[])

    # ----------------------
    # Apply filters
    # ----------------------
    start_date, end_date = date_range
    # normalize start/end to timestamps
    if isinstance(start_date, datetime):
        start_date = pd.to_datetime(start_date).normalize()
    else:
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, datetime):
        end_date = pd.to_datetime(end_date).normalize()
    else:
        end_date = pd.to_datetime(end_date)

    # build marketing mask
    mask_marketing = (
        (marketing['date'] >= start_date)
        & (marketing['date'] <= end_date)
        & (marketing['channel'].isin(selected_channels))
    )
    if selected_states:
        mask_marketing &= marketing['state'].astype(str).isin(selected_states)
    if selected_tactics:
        mask_marketing &= marketing['tactic'].astype(str).isin(selected_tactics)
    if selected_campaigns:
        mask_marketing &= marketing['campaign'].astype(str).isin(selected_campaigns)

    marketing_filt = marketing[mask_marketing].copy()
    if marketing_filt.empty:
        st.warning("No marketing rows match the filter. Please adjust filters.")

    # Re-aggregate filtered marketing
    marketing_filt_daily = (
        marketing_filt
        .groupby(['date', 'channel'], as_index=False)
        .agg({'impressions': 'sum', 'clicks': 'sum', 'spend': 'sum', 'attributed_revenue': 'sum'})
    )

    marketing_filt_total = (
        marketing_filt_daily
        .groupby('date', as_index=False)
        .agg({'impressions': 'sum', 'clicks': 'sum', 'spend': 'sum', 'attributed_revenue': 'sum'})
        .rename(columns=lambda x: f"marketing_{x}" if x != 'date' else x)
    )

    # filter business data to the selected date range
    biz_filt = biz[(biz['date'] >= start_date) & (biz['date'] <= end_date)].copy()

    # merge business + filtered marketing totals for downstream metrics & visuals
    df_filt = pd.merge(biz_filt, marketing_filt_total, on='date', how='left').fillna(0)
    # compute metrics for filtered df
    df_filt['ctr'] = np.where(df_filt['marketing_impressions'] > 0, df_filt['marketing_clicks'] / df_filt['marketing_impressions'], np.nan)
    df_filt['cpc'] = np.where(df_filt['marketing_clicks'] > 0, df_filt['marketing_spend'] / df_filt['marketing_clicks'], np.nan)
    df_filt['roas'] = np.where(df_filt['marketing_spend'] > 0, df_filt['marketing_attributed_revenue'] / df_filt['marketing_spend'], np.nan)
    df_filt['marketing_pct_of_revenue'] = np.where(df_filt['revenue'] > 0, df_filt['marketing_attributed_revenue'] / df_filt['revenue'], 0)

    # ----------------------
    # KPI cards
    # ----------------------
    st.subheader("Key Metrics")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    total_spend = df_filt['marketing_spend'].sum()
    total_attr_rev = df_filt['marketing_attributed_revenue'].sum()
    avg_roas = np.nanmean(df_filt['roas'].replace([np.inf, -np.inf], np.nan))
    avg_cpc = np.nanmean(df_filt['cpc'].replace([np.inf, -np.inf], np.nan))
    total_orders = df_filt['orders'].sum()

    kpi1.metric("Total Spend", f"${total_spend:,.0f}")
    kpi2.metric("Attributed Revenue", f"${total_attr_rev:,.0f}")
    kpi3.metric("Avg ROAS", f"{avg_roas:.2f}" if not np.isnan(avg_roas) else "N/A")
    kpi4.metric("Avg CPC", f"${avg_cpc:.2f}" if not np.isnan(avg_cpc) else "N/A")
    kpi5.metric("Total Orders (business)", f"{int(total_orders):,}")

    # ----------------------
    # Time series: spend vs revenue
    # ----------------------
    st.markdown("### Spend vs Revenue (daily)")
    ts = df_filt.sort_values('date')
    if not ts.empty:
        fig = px.line(
            ts,
            x='date',
            y=['marketing_spend', 'revenue', 'marketing_attributed_revenue'],
            labels={'value': 'USD', 'date': 'Date'},
            title='Daily Marketing Spend, Attributed Revenue and Business Revenue'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to show in time series for selected filters/dates.")

    # ----------------------
    # Channel comparison
    # ----------------------
    st.markdown("### Channel-level summary (filtered)")
    channel_summary = (
        marketing_filt.groupby('channel', as_index=False)
        .agg({'impressions': 'sum', 'clicks': 'sum', 'spend': 'sum', 'attributed_revenue': 'sum'})
    )

    # Estimate channel orders by allocating business orders by proportion of attributed revenue per date
    if not marketing_filt.empty and not biz_filt.empty:
        # compute daily shares then aggregate
        daily = (
            marketing_filt
            .groupby(['date', 'channel'], as_index=False)
            .agg({'attributed_revenue': 'sum', 'spend': 'sum', 'clicks': 'sum', 'impressions': 'sum'})
        )
        daily_total = (
            daily
            .groupby('date', as_index=False)
            .agg({'attributed_revenue': 'sum'})
            .rename(columns={'attributed_revenue': 'total_attr_rev'})
        )

        daily = daily.merge(daily_total, on='date', how='left')
        daily = daily.merge(biz_filt[['date', 'orders']], on='date', how='left').fillna(0)

        # channel estimated orders on each date
        daily['channel_est_orders'] = np.where(
            daily['total_attr_rev'] > 0,
            (daily['attributed_revenue'] / daily['total_attr_rev']) * daily['orders'],
            0
        )

        channel_orders = (
            daily
            .groupby('channel', as_index=False)
            .agg({
                'channel_est_orders': 'sum',
                'spend': 'sum',
                'attributed_revenue': 'sum',
                'clicks': 'sum',
                'impressions': 'sum'
            })
            .rename(columns={'channel_est_orders': 'est_orders'})
        )

        channel_orders['est_cpa'] = np.where(channel_orders['est_orders'] > 0, channel_orders['spend'] / channel_orders['est_orders'], np.nan)
        channel_orders['roas'] = np.where(channel_orders['spend'] > 0, channel_orders['attributed_revenue'] / channel_orders['spend'], np.nan)
    else:
        channel_orders = pd.DataFrame(columns=['channel', 'est_orders', 'spend', 'attributed_revenue', 'clicks', 'impressions', 'est_cpa', 'roas'])

    # Display table and charts
    if not channel_summary.empty:
        st.dataframe(
            channel_orders.sort_values('spend', ascending=False).style.format({
                'spend': '${:,.2f}',
                'attributed_revenue': '${:,.2f}',
                'est_cpa': '${:,.2f}',
                'roas': '{:.2f}',
                'est_orders': '{:.0f}'
            }),
            height=240,
        )

        fig2 = px.bar(
            channel_orders.sort_values('spend', ascending=False),
            x='channel',
            y=['spend', 'attributed_revenue'],
            title="Spend vs Attributed Revenue by Channel (filtered)",
            barmode='group'
        )
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.bar(channel_orders.sort_values('est_cpa'), x='channel', y='est_cpa', title='Estimated CPA by Channel')
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No channel-level data to display for selected filters.")

    # ----------------------
    # Top campaigns
    # ----------------------
    st.markdown("### Top campaigns (by spend / attributed revenue)")
    if selected_campaigns:
        campaign_mask = (
            marketing['campaign'].astype(str).isin(selected_campaigns)
            & (marketing['date'] >= start_date)
            & (marketing['date'] <= end_date)
            & (marketing['channel'].isin(selected_channels))
        )
    else:
        campaign_mask = (
            (marketing['date'] >= start_date)
            & (marketing['date'] <= end_date)
            & (marketing['channel'].isin(selected_channels))
        )

    campaigns_agg = (
        marketing[campaign_mask]
        .groupby(['channel', 'campaign'], as_index=False)
        .agg({'impressions': 'sum', 'clicks': 'sum', 'spend': 'sum', 'attributed_revenue': 'sum'})
        .sort_values('spend', ascending=False)
    )

    if not campaigns_agg.empty:
        campaigns_agg['cpc'] = np.where(campaigns_agg['clicks'] > 0, campaigns_agg['spend'] / campaigns_agg['clicks'], np.nan)
        campaigns_agg['roas'] = np.where(campaigns_agg['spend'] > 0, campaigns_agg['attributed_revenue'] / campaigns_agg['spend'], np.nan)

        st.dataframe(campaigns_agg.head(30).style.format({
            'spend': '${:,.2f}',
            'attributed_revenue': '${:,.2f}',
            'cpc': '${:,.2f}',
            'roas': '{:.2f}'
        }), height=350)
    else:
        st.info("No campaign data for selected filters.")

    # ----------------------
    # Export filtered data
    # ----------------------
    st.markdown("### Export data")
    export_df = marketing_filt.copy()
    export_df = export_df.sort_values(['date', 'channel'])
    if not export_df.empty:
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download filtered marketing rows as CSV",
            data=csv,
            file_name=f"marketing_filtered_{start_date.date()}_{end_date.date()}.csv",
            mime="text/csv",
        )
    else:
        st.info("No rows available to export for selected filters.")

    # ----------------------
    # Notes & assumptions
    # ----------------------
    st.markdown(
        """
    ### Notes & assumptions
    - `attributed_revenue` in the marketing files is used as the marketing-attributed revenue. The business `revenue` column is the total daily revenue (may include non-marketing revenue).
    - Because orders are not campaign-attributed in the dataset, the dashboard estimates channel-level orders by allocating daily business orders to channels in proportion to daily `attributed_revenue`. This is an approximation — if you have a deterministic mapping of orders to campaigns, replace the estimation logic.
    - All numeric conversions use safe coercion; missing numeric values are treated as 0.
    - Use the sidebar to filter date range, channels, states, tactics, or campaigns.
        """
    )

else:
    st.warning("Please upload all four CSV files.")

