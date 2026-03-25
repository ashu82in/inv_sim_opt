import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(layout="wide")
st.title("Inventory Policy Simulator")

# =========================================================
# SESSION STATE
# =========================================================
if "demand_sequence" not in st.session_state:
    st.session_state.demand_sequence = None

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Inventory Inputs")

opening_balance = st.sidebar.number_input("Opening Balance", value=500)
avg_demand = st.sidebar.number_input("Average Demand", value=25)
cov = st.sidebar.number_input("Coefficient of Variation", value=0.8)
lead_time = st.sidebar.number_input("Lead Time", value=3)

reorder_point_input = st.sidebar.number_input("Reorder Point", value=200)

order_qty = st.sidebar.number_input("Order Quantity", value=300)
unit_value = st.sidebar.number_input("Value Per Unit", value=100)
holding_cost_percent = st.sidebar.number_input("Holding Cost (%)", value=20.0)
ordering_cost = st.sidebar.number_input("Ordering Cost", value=500)
num_days = st.sidebar.slider("Simulation Days", 100, 2000, 365)

holding_cost_rate = holding_cost_percent / 100
std_demand = avg_demand * cov

# =========================================================
# RESET DEMAND (NO RERUN)
# =========================================================
if st.button("Reset Demand Scenario"):
    st.session_state.demand_sequence = None

# =========================================================
# DEMAND
# =========================================================
if st.session_state.demand_sequence is None:
    st.session_state.demand_sequence = np.maximum(
        0,
        np.random.normal(avg_demand, std_demand, num_days)
    ).round()

demand = st.session_state.demand_sequence

# =========================================================
# SIMULATION FUNCTION
# =========================================================
def run_simulation(demand, reorder_point, order_qty):

    inventory = opening_balance
    pipeline_orders = []
    data = []

    for day in range(num_days):

        shipment_received = 0
        for order in pipeline_orders.copy():
            if order[0] == day:
                shipment_received += order[1]
                pipeline_orders.remove(order)

        opening = inventory
        inventory += shipment_received

        demand_today = demand[day]
        inventory -= demand_today

        if inventory < 0:
            inventory = 0

        pipeline_qty = sum(qty for arrival, qty in pipeline_orders)
        inventory_position = opening - demand_today + shipment_received + pipeline_qty

        new_order = 0
        if inventory_position < reorder_point:
            new_order = order_qty
            pipeline_orders.append((day + lead_time, order_qty))

        closing = inventory
        closing_with_pipeline = closing + sum(qty for arrival, qty in pipeline_orders)

        data.append([
            opening, demand_today, shipment_received, pipeline_qty,
            inventory_position, new_order, closing, closing_with_pipeline
        ])

    return pd.DataFrame(data, columns=[
        "Opening Balance","Demand","Shipment Received","Pipeline Order",
        "Inventory Position","New Order","Closing Balance",
        "Closing Balance Including Pipeline"
    ])

# =========================================================
# POLICY INPUT
# =========================================================
st.subheader("Policy Input")

use_service_level = st.checkbox(
    "Use Service Level to Calculate Reorder Point",
    key="use_sl"
)

service_level_input = st.slider(
    "Target Service Level",
    0.80, 0.99, 0.95,
    key="sl_value",
    disabled=not use_service_level
)

reorder_point = reorder_point_input

if use_service_level:
    z = norm.ppf(service_level_input)
    mean_lt = avg_demand * lead_time
    std_lt = std_demand * np.sqrt(lead_time)
    reorder_point = int(mean_lt + z * std_lt)
    st.success(f"Auto Reorder Point: {reorder_point}")
else:
    st.info(f"Manual Reorder Point: {reorder_point}")

# =========================================================
# RUN SIMULATION
# =========================================================
df = run_simulation(demand, reorder_point, order_qty)
df["Date"] = pd.date_range(start="2024-01-01", periods=num_days)

# =========================================================
# KPI CALCULATIONS
# =========================================================
stockout_days = (df["Closing Balance"] == 0).sum()

df["Blocked Working Capital"] = df["Inventory Position"] * unit_value
average_working_capital = df["Blocked Working Capital"].mean()

average_inventory = df["Closing Balance Including Pipeline"].mean()
average_age_inventory = average_inventory / df["Demand"].mean()

df["Inventory Value"] = df["Closing Balance Including Pipeline"] * unit_value
df["Holding Cost"] = df["Inventory Value"] * holding_cost_rate / 365

total_holding_cost = df["Holding Cost"].sum()
total_ordering_cost = (df["New Order"] > 0).sum() * ordering_cost
total_inventory_cost = total_holding_cost + total_ordering_cost

# =========================================================
# INVENTORY AGE
# =========================================================
df["Inventory Age"] = df["Closing Balance Including Pipeline"] / df["Demand"].replace(0, np.nan)
df["Inventory Age"] = df["Inventory Age"].fillna(0)

# =========================================================
# KPI DISPLAY
# =========================================================
st.subheader("Inventory KPIs")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Stockout Days", stockout_days)
c2.metric("Average Age of Inventory", round(average_age_inventory,1))
c3.metric("Average Inventory", round(average_inventory,0))
c4.metric("Avg Working Capital", round(average_working_capital,0))

# =========================================================
# INVENTORY CHART
# =========================================================
st.subheader("Inventory Behaviour")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance"], name="Closing"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance Including Pipeline"], name="Position"))

fig.add_hline(y=reorder_point, line_dash="dash")
fig.add_hrect(y0=0, y1=reorder_point*0.5, fillcolor="red", opacity=0.1)
fig.add_hrect(y0=reorder_point*0.5, y1=reorder_point, fillcolor="yellow", opacity=0.1)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# WORKING CAPITAL
# =========================================================
st.subheader("Blocked Working Capital")

fig_wc = px.line(df, x="Date", y="Blocked Working Capital")
st.plotly_chart(fig_wc, use_container_width=True)

# =========================================================
# INVENTORY AGE CHART
# =========================================================
st.subheader("Inventory Age")

fig_age = px.line(df, x="Date", y="Inventory Age")
st.plotly_chart(fig_age, use_container_width=True)

# =========================================================
# AGING BUCKETS
# =========================================================
st.subheader("Aging Buckets")

conditions = [
    (df["Inventory Age"] <= 30),
    (df["Inventory Age"] <= 60),
    (df["Inventory Age"] <= 90),
    (df["Inventory Age"] > 90)
]

choices = ["0-30", "31-60", "61-90", "90+"]

df["Age Bucket"] = np.select(conditions, choices)

bucket_df = df.groupby(["Date", "Age Bucket"])["Inventory Value"].sum().reset_index()

fig_bucket = px.bar(bucket_df, x="Date", y="Inventory Value", color="Age Bucket")
st.plotly_chart(fig_bucket, use_container_width=True)

# =========================================================
# 🚨 DEAD STOCK
# =========================================================
st.subheader("🚨 Dead Stock Analysis")

df["Dead Stock Value"] = np.where(
    df["Inventory Age"] > 90,
    df["Closing Balance Including Pipeline"] * unit_value,
    0
)

dead_stock_value = df["Dead Stock Value"].sum()
total_inventory_value = df["Inventory Value"].sum()

dead_stock_percent = (dead_stock_value / total_inventory_value * 100) if total_inventory_value > 0 else 0
dead_stock_days = (df["Dead Stock Value"] > 0).sum()

# ALERT
if dead_stock_percent > 30:
    st.error(f"🚨 CRITICAL: ₹{round(dead_stock_value)} ({round(dead_stock_percent,1)}%) dead stock")
elif dead_stock_percent > 15:
    st.warning(f"⚠️ ₹{round(dead_stock_value)} ({round(dead_stock_percent,1)}%) dead stock")
elif dead_stock_value > 0:
    st.info(f"ℹ️ ₹{round(dead_stock_value)} ({round(dead_stock_percent,1)}%) dead stock")
else:
    st.success("✅ No dead stock")

# KPIs
k1,k2,k3 = st.columns(3)
k1.metric("Dead Stock Value", round(dead_stock_value))
k2.metric("Dead Stock %", f"{round(dead_stock_percent,1)}%")
k3.metric("Days with Dead Stock", dead_stock_days)

# Trend
fig_dead = px.line(df, x="Date", y="Dead Stock Value")
st.plotly_chart(fig_dead, use_container_width=True)

# =========================================================
# DEMAND
# =========================================================
st.subheader("Demand Distribution")
st.plotly_chart(px.histogram(df, x="Demand"), use_container_width=True)

# =========================================================
# DATA
# =========================================================
st.subheader("Simulation Data")
st.dataframe(df)
