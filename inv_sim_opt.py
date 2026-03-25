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
# SIDEBAR INPUTS
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
# RESET DEMAND
# =========================================================
if st.button("Reset Demand Scenario"):
    st.session_state.demand_sequence = None

# =========================================================
# DEMAND
# =========================================================
if st.session_state.demand_sequence is None:
    st.session_state.demand_sequence = np.maximum(
        0, np.random.normal(avg_demand, std_demand, num_days)
    ).round()

demand = st.session_state.demand_sequence

# =========================================================
# FIFO SIMULATION WITH AGING
# =========================================================
def run_simulation(demand, reorder_point, order_qty):

    inventory_layers = [{"qty": opening_balance, "age": 0}]
    pipeline_orders = []
    data = []
    aging_data = []

    for day in range(num_days):

        # Age all inventory
        for layer in inventory_layers:
            layer["age"] += 1

        # Receive orders
        for order in pipeline_orders.copy():
            if order[0] == day:
                inventory_layers.append({"qty": order[1], "age": 0})
                pipeline_orders.remove(order)

        # Demand consumption (FIFO)
        remaining = demand[day]

        while remaining > 0 and inventory_layers:
            if inventory_layers[0]["qty"] <= remaining:
                remaining -= inventory_layers[0]["qty"]
                inventory_layers.pop(0)
            else:
                inventory_layers[0]["qty"] -= remaining
                remaining = 0

        total_inventory = sum(l["qty"] for l in inventory_layers)

        pipeline_qty = sum(q for _, q in pipeline_orders)
        inventory_position = total_inventory + pipeline_qty

        new_order = 0
        if inventory_position < reorder_point:
            new_order = order_qty
            pipeline_orders.append((day + lead_time, order_qty))

        data.append([
            demand[day], total_inventory, inventory_position, new_order
        ])

        # Aging buckets
        bucket = {"0-30":0, "31-60":0, "61-90":0, "90+":0}

        for l in inventory_layers:
            age, qty = l["age"], l["qty"]

            if age <= 30:
                bucket["0-30"] += qty
            elif age <= 60:
                bucket["31-60"] += qty
            elif age <= 90:
                bucket["61-90"] += qty
            else:
                bucket["90+"] += qty

        bucket["Day"] = day
        aging_data.append(bucket)

    df = pd.DataFrame(data, columns=[
        "Demand","Closing Balance","Inventory Position","New Order"
    ])

    aging_df = pd.DataFrame(aging_data)

    return df, aging_df

# =========================================================
# POLICY INPUT
# =========================================================
st.subheader("Policy Input")

use_service_level = st.checkbox("Use Service Level", key="use_sl")

service_level_input = st.slider(
    "Target Service Level", 0.80, 0.99, 0.95,
    key="sl_value",
    disabled=not use_service_level
)

reorder_point = reorder_point_input

if use_service_level:
    z = norm.ppf(service_level_input)
    reorder_point = int(avg_demand*lead_time + z*std_demand*np.sqrt(lead_time))
    st.success(f"Auto Reorder Point: {reorder_point}")
else:
    st.info(f"Manual Reorder Point: {reorder_point}")

# =========================================================
# RUN SIMULATION
# =========================================================
df, aging_df = run_simulation(demand, reorder_point, order_qty)
df["Date"] = pd.date_range("2024-01-01", periods=num_days)
aging_df["Date"] = df["Date"]

# =========================================================
# KPI CALCULATIONS (FULL RESTORED)
# =========================================================

df["Blocked Working Capital"] = df["Inventory Position"] * unit_value
df["Inventory Value"] = df["Inventory Position"] * unit_value

stockout_days = (df["Closing Balance"] == 0).sum()

average_inventory = df["Inventory Position"].mean()
average_age_inventory = average_inventory / df["Demand"].mean()

average_working_capital = df["Blocked Working Capital"].mean()

min_inventory = df["Closing Balance"].min()
max_inventory = df["Closing Balance"].max()

min_wc = df["Blocked Working Capital"].min()
max_wc = df["Blocked Working Capital"].max()

df["Holding Cost"] = df["Inventory Value"] * holding_cost_rate / 365

total_holding_cost = df["Holding Cost"].sum()
number_of_orders = (df["New Order"] > 0).sum()
total_ordering_cost = number_of_orders * ordering_cost
total_inventory_cost = total_holding_cost + total_ordering_cost

annual_demand = avg_demand * 365
holding_cost_per_unit = unit_value * holding_cost_rate
eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)

# =========================================================
# KPI DISPLAY
# =========================================================

st.subheader("Inventory KPIs")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Stockout Days", stockout_days)
c2.metric("Average Age of Inventory", round(average_age_inventory,1))
c3.metric("Average Inventory", round(average_inventory,0))
c4.metric("Avg Working Capital", round(average_working_capital,0))

st.subheader("Inventory Range")

r1, r2, r3, r4 = st.columns(4)
r1.metric("Minimum Inventory", round(min_inventory,0))
r2.metric("Maximum Inventory", round(max_inventory,0))
r3.metric("Minimum Working Capital", round(min_wc,0))
r4.metric("Maximum Working Capital", round(max_wc,0))

st.subheader("Inventory Cost Metrics")

cc1, cc2, cc3 = st.columns(3)
cc1.metric("Total Holding Cost", round(total_holding_cost,0))
cc2.metric("Total Ordering Cost", round(total_ordering_cost,0))
cc3.metric("Total Inventory Cost", round(total_inventory_cost,0))

st.subheader("EOQ")

e1, e2 = st.columns(2)
e1.metric("Economic Order Quantity", round(eoq,0))
e2.metric("Selected Order Quantity", order_qty)

# =========================================================
# INVENTORY BEHAVIOUR (WITH RED ZONE)
# =========================================================

st.subheader("Inventory Behaviour")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance"], name="Inventory"))

fig.add_hline(y=reorder_point, line_dash="dash")

fig.add_hrect(y0=0, y1=reorder_point*0.5, fillcolor="red", opacity=0.1)
fig.add_hrect(y0=reorder_point*0.5, y1=reorder_point, fillcolor="yellow", opacity=0.1)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# AGING BUCKETS (REALISTIC)
# =========================================================

st.subheader("Aging Buckets")

for col in ["0-30","31-60","61-90","90+"]:
    aging_df[col] *= unit_value

aging_melt = aging_df.melt(
    id_vars=["Date"],
    value_vars=["0-30","31-60","61-90","90+"],
    var_name="Bucket",
    value_name="Value"
)

fig_bucket = px.bar(
    aging_melt,
    x="Date",
    y="Value",
    color="Bucket",
    color_discrete_map={
        "0-30":"#ADD8E6",
        "31-60":"#6495ED",
        "61-90":"#FFA07A",
        "90+":"#FF0000"
    }
)

st.plotly_chart(fig_bucket, use_container_width=True)

# =========================================================
# DEAD STOCK
# =========================================================

st.subheader("Dead Stock Analysis")

dead_stock = aging_df["90+"].sum()
total = aging_df[["0-30","31-60","61-90","90+"]].sum().sum()

pct = (dead_stock/total*100) if total>0 else 0

if pct > 30:
    st.error(f"🚨 ₹{round(dead_stock)} dead stock ({round(pct,1)}%)")
elif pct > 15:
    st.warning(f"⚠️ ₹{round(dead_stock)} dead stock ({round(pct,1)}%)")
else:
    st.success("✅ Healthy inventory")

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
