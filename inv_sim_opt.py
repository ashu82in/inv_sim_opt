import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(layout="wide")

st.title("Inventory Policy Simulator")

# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.header("Inventory Inputs")

opening_balance = st.sidebar.number_input("Opening Balance", value=500)
avg_demand = st.sidebar.number_input("Average Demand", value=25)
cov = st.sidebar.number_input("Coefficient of Variation", value=0.8)
lead_time = st.sidebar.number_input("Lead Time (Days)", value=3)

reorder_point_input = st.sidebar.number_input(
    "Reorder Point",
    value=200,
    key="rp_sidebar"
)

order_qty = st.sidebar.number_input("Order Quantity", value=300)
unit_value = st.sidebar.number_input("Value Per Unit", value=100)
holding_cost_percent = st.sidebar.number_input("Holding Cost (%)", value=20.0)
ordering_cost = st.sidebar.number_input("Ordering Cost", value=500)
num_days = st.sidebar.slider("Simulation Days", 100, 2000, 365)

st.sidebar.caption("👉 Used when Service Level checkbox is OFF")

holding_cost_rate = holding_cost_percent / 100
std_demand = avg_demand * cov

# =========================================================
# DEMAND STATE
# =========================================================

if "demand_sequence" not in st.session_state:
    st.session_state.demand_sequence = None

if st.session_state.demand_sequence is None:
    st.session_state.demand_sequence = np.maximum(
        0,
        np.random.normal(avg_demand, std_demand, num_days)
    ).round()

demand = st.session_state.demand_sequence

# =========================================================
# FUNCTIONS
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

    df = pd.DataFrame(data, columns=[
        "Opening Balance","Demand","Shipment Received","Pipeline Order",
        "Inventory Position","New Order","Closing Balance",
        "Closing Balance Including Pipeline"
    ])

    return df


def run_simulation_metrics(demand, reorder_point, order_qty):
    inventory = opening_balance
    pipeline_orders = []

    stockout_days = 0
    holding_cost_total = 0
    orders_count = 0

    for day in range(num_days):

        shipment_received = 0
        for order in pipeline_orders.copy():
            if order[0] == day:
                shipment_received += order[1]
                pipeline_orders.remove(order)

        inventory += shipment_received

        demand_today = demand[day]
        inventory -= demand_today

        if inventory < 0:
            inventory = 0
            stockout_days += 1

        pipeline_qty = sum(qty for arrival, qty in pipeline_orders)
        inventory_position = inventory + pipeline_qty

        if inventory_position < reorder_point:
            pipeline_orders.append((day + lead_time, order_qty))
            orders_count += 1

        closing_with_pipeline = inventory + sum(qty for arrival, qty in pipeline_orders)

        inventory_value = closing_with_pipeline * unit_value
        holding_cost_total += inventory_value * holding_cost_rate / 365

    total_cost = holding_cost_total + orders_count * ordering_cost

    return stockout_days, total_cost


# =========================================================
# MAIN TAB
# =========================================================

tab1 = st.tabs(["Simulation"])[0]

with tab1:

    # RESET BUTTON (TOP)
    col_btn, _ = st.columns([1,5])
    with col_btn:
        if st.button("Reset Demand Scenario"):
            st.session_state.demand_sequence = None
            st.rerun()

    # =========================================================
    # POLICY INPUT (CHECKBOX BASED - FINAL)
    # =========================================================

    st.subheader("Policy Input")

    use_service_level = st.checkbox(
        "Use Service Level to Calculate Reorder Point",
        value=st.session_state.get("use_sl", True),
        key="use_sl"
    )

    service_level_input = st.slider(
        "Target Service Level",
        0.80, 0.99, 0.95,
        key="sl_value",
        disabled=not use_service_level
    )

    # SINGLE SOURCE OF TRUTH
    reorder_point = reorder_point_input

    if use_service_level:
        z = norm.ppf(service_level_input)
        mean_lt = avg_demand * lead_time
        std_lt = std_demand * np.sqrt(lead_time)
        reorder_point = int(mean_lt + z * std_lt)
        st.success(f"Auto Reorder Point: {reorder_point}")
    else:
        st.info(f"Manual Reorder Point (Sidebar): {reorder_point}")

    # =========================================================
    # RUN SIMULATION
    # =========================================================

    df = run_simulation(demand, reorder_point, order_qty)
    df["Date"] = pd.date_range(start="2024-01-01", periods=num_days)

    # KPIs
    stockout_days = (df["Closing Balance"] == 0).sum()
    avg_inventory = df["Closing Balance Including Pipeline"].mean()
    avg_age = avg_inventory / df["Demand"].mean()

    df["Blocked Working Capital"] = df["Inventory Position"] * unit_value
    avg_wc = df["Blocked Working Capital"].mean()

    min_inventory = df["Closing Balance"].min()
    max_inventory = df["Closing Balance"].max()

    min_wc = df["Blocked Working Capital"].min()
    max_wc = df["Blocked Working Capital"].max()

    df["Inventory Value"] = df["Closing Balance Including Pipeline"] * unit_value
    df["Holding Cost"] = df["Inventory Value"] * holding_cost_rate / 365

    total_holding_cost = df["Holding Cost"].sum()
    total_ordering_cost = (df["New Order"] > 0).sum() * ordering_cost
    total_inventory_cost = total_holding_cost + total_ordering_cost

    # EOQ
    annual_demand = avg_demand * 365
    holding_cost_per_unit = unit_value * holding_cost_rate
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)

    _, cost_eoq = run_simulation_metrics(demand, reorder_point, int(eoq))

    # KPI DISPLAY
    st.subheader("Inventory KPIs")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Stockout Days", stockout_days)
    c2.metric("Average Age of Inventory", round(avg_age,1))
    c3.metric("Average Inventory", round(avg_inventory,0))
    c4.metric("Avg Working Capital", round(avg_wc,0))

    st.subheader("Inventory Range")

    r1,r2,r3,r4 = st.columns(4)
    r1.metric("Minimum Inventory", round(min_inventory,0))
    r2.metric("Maximum Inventory", round(max_inventory,0))
    r3.metric("Minimum Working Capital", round(min_wc,0))
    r4.metric("Maximum Working Capital", round(max_wc,0))

    st.subheader("Inventory Cost Metrics")

    cc1,cc2,cc3 = st.columns(3)
    cc1.metric("Total Holding Cost", round(total_holding_cost,0))
    cc2.metric("Total Ordering Cost", round(total_ordering_cost,0))
    cc3.metric("Total Inventory Cost", round(total_inventory_cost,0))

    st.subheader("EOQ")

    e1,e2 = st.columns(2)
    e1.metric("Economic Order Quantity", round(eoq,0))
    e2.metric("Selected Order Quantity", order_qty)

    st.subheader("Cost Comparison")

    k1,k2,k3 = st.columns(3)
    k1.metric("Cost with Current Policy", round(total_inventory_cost,0))
    k2.metric("Cost with EOQ", round(cost_eoq,0))
    k3.metric("Savings Using EOQ", round(total_inventory_cost-cost_eoq,0))

    # INSIGHTS
    st.subheader("🔍 Business Insights")

    df_eoq = run_simulation(demand, reorder_point, int(eoq))
    eoq_inventory = df_eoq["Closing Balance Including Pipeline"].mean()
    df_eoq["Blocked Working Capital"] = df_eoq["Inventory Position"] * unit_value
    eoq_wc = df_eoq["Blocked Working Capital"].mean()

    inventory_diff = avg_inventory - eoq_inventory
    wc_diff = avg_wc - eoq_wc

    if inventory_diff > 0:
        st.warning(f"⚠️ Excess inventory ~{round(inventory_diff,0)} units (~₹{round(wc_diff,0)})")
    elif inventory_diff < 0:
        st.error(f"⚠️ Understocking risk ~{round(abs(inventory_diff),0)} units")
    else:
        st.success("✅ Inventory policy is balanced")

    # CHART
    st.subheader("Inventory Behaviour")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance"], name="Closing"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance Including Pipeline"], name="Position"))

    fig.add_hline(y=reorder_point, line_dash="dash")
    fig.add_hrect(y0=0, y1=reorder_point*0.5, fillcolor="red", opacity=0.1)
    fig.add_hrect(y0=reorder_point*0.5, y1=reorder_point, fillcolor="yellow", opacity=0.1)

    st.plotly_chart(fig, use_container_width=True)

    # DEMAND HISTOGRAM
    st.subheader("Demand Distribution")
    st.plotly_chart(px.histogram(df, x="Demand"), use_container_width=True)

    # DATA TABLE
    st.subheader("Simulation Data")
    st.dataframe(df)
