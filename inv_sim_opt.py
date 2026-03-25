import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(layout="wide")

st.title("Inventory Decision Engine")

# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.header("Inventory Inputs")

opening_balance = st.sidebar.number_input("Opening Balance", value=500)
avg_demand = st.sidebar.number_input("Average Demand", value=25)
cov = st.sidebar.number_input("Coefficient of Variation", value=0.8)
lead_time = st.sidebar.number_input("Lead Time (Days)", value=3)
reorder_point_input = st.sidebar.number_input("Manual Reorder Point", value=200)
order_qty = st.sidebar.number_input("Order Quantity", value=300)
unit_value = st.sidebar.number_input("Value Per Unit", value=100)
holding_cost_percent = st.sidebar.number_input("Holding Cost (%)", value=20.0)
ordering_cost = st.sidebar.number_input("Ordering Cost", value=500)
num_days = st.sidebar.slider("Simulation Days", 100, 2000, 365)

holding_cost_rate = holding_cost_percent / 100
std_demand = avg_demand * cov

st.sidebar.caption("👉 Click below to simulate a new demand pattern")

# =========================================================
# DEMAND STATE
# =========================================================

if "demand_sequence" not in st.session_state:
    st.session_state.demand_sequence = None

if st.sidebar.button("Reset Demand Scenario"):
    st.session_state.demand_sequence = None

if st.session_state.demand_sequence is None:
    st.session_state.demand_sequence = np.maximum(
        0,
        np.random.normal(avg_demand, std_demand, num_days)
    ).round()

demand = st.session_state.demand_sequence

# =========================================================
# CORE FUNCTIONS
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
        "Opening Balance", "Demand", "Shipment Received",
        "Pipeline Order", "Inventory Position", "New Order",
        "Closing Balance", "Closing Balance Including Pipeline"
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
        holding_cost_today = inventory_value * holding_cost_rate / 365

        holding_cost_total += holding_cost_today

    total_cost = holding_cost_total + orders_count * ordering_cost

    return stockout_days, total_cost

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "Single Simulation",
    "Scenario Analysis",
    "Auto Optimization"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.subheader("Policy Input")

    use_service_level = st.toggle(
        "Use Service Level to Calculate Reorder Point",
        value=True,
        key="toggle_tab1"
    )

    service_level_input = st.slider(
        "Target Service Level",
        0.80, 0.99, 0.95,
        key="sl_tab1"
    )

    if use_service_level:
        z = norm.ppf(service_level_input)
        mean_lt = avg_demand * lead_time
        std_lt = std_demand * np.sqrt(lead_time)
        reorder_point = int(mean_lt + z * std_lt)
        st.info(f"Calculated Reorder Point: {reorder_point}")
    else:
        reorder_point = reorder_point_input

    df = run_simulation(demand, reorder_point, order_qty)

    # KPIs
    stockout_days = (df["Closing Balance"] == 0).sum()
    avg_inventory = df["Closing Balance Including Pipeline"].mean()
    avg_age = avg_inventory / df["Demand"].mean()

    df["Blocked Working Capital"] = df["Inventory Position"] * unit_value
    avg_wc = df["Blocked Working Capital"].mean()

    df["Inventory Value"] = df["Closing Balance Including Pipeline"] * unit_value
    df["Holding Cost"] = df["Inventory Value"] * holding_cost_rate / 365

    total_cost = df["Holding Cost"].sum() + (df["New Order"] > 0).sum() * ordering_cost

    st.subheader("KPIs")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Stockouts", stockout_days)
    c2.metric("Avg Age", round(avg_age,1))
    c3.metric("Avg Inventory", round(avg_inventory,0))
    c4.metric("Avg WC", round(avg_wc,0))

    st.metric("Total Cost", round(total_cost,0))

    st.subheader("Service Level")
    achieved_sl = 1 - (stockout_days > 0)
    st.metric("Achieved Service Level", round(achieved_sl,2))

    st.plotly_chart(px.line(df, y="Closing Balance"), use_container_width=True)

# =========================================================
# TAB 2
# =========================================================

with tab2:

    num_simulations = st.number_input(
        "Simulations",
        value=100,
        key="sim_tab2"
    )

    results = []

    for _ in range(int(num_simulations)):

        demand_sim = np.maximum(
            0,
            np.random.normal(avg_demand, std_demand, num_days)
        ).round()

        df_sim = run_simulation(demand_sim, reorder_point_input, order_qty)

        results.append({
            "Stockout Days": (df_sim["Closing Balance"] == 0).sum(),
            "Avg Inventory": df_sim["Closing Balance Including Pipeline"].mean()
        })

    results_df = pd.DataFrame(results)

    st.plotly_chart(px.histogram(results_df, x="Stockout Days"))
    st.plotly_chart(px.histogram(results_df, x="Avg Inventory"))

# =========================================================
# TAB 3
# =========================================================

with tab3:

    target_service_level = st.slider(
        "Target Service Level",
        0.80, 0.99, 0.95,
        key="sl_tab3"
    )

    rp_min = st.number_input("Min RP", value=100, key="rp_min")
    rp_max = st.number_input("Max RP", value=400, key="rp_max")

    oq_min = st.number_input("Min OQ", value=100, key="oq_min")
    oq_max = st.number_input("Max OQ", value=500, key="oq_max")

    step = st.number_input("Step", value=50, key="step")
    num_simulations = st.number_input("Simulations", value=50, key="sim_tab3")

    if st.button("Run Optimization"):

        results = []

        for rp in range(int(rp_min), int(rp_max)+1, int(step)):
            for oq in range(int(oq_min), int(oq_max)+1, int(step)):

                stockouts = []
                costs = []

                for _ in range(int(num_simulations)):

                    demand_sim = np.maximum(
                        0,
                        np.random.normal(avg_demand, std_demand, num_days)
                    ).round()

                    so, cost = run_simulation_metrics(demand_sim, rp, oq)

                    stockouts.append(so)
                    costs.append(cost)

                service_level = 1 - (np.array(stockouts) > 0).mean()

                results.append({
                    "RP": rp,
                    "OQ": oq,
                    "Service Level": service_level,
                    "Cost": np.mean(costs)
                })

        results_df = pd.DataFrame(results)

        feasible = results_df[results_df["Service Level"] >= target_service_level]

        if len(feasible) > 0:
            best = feasible.sort_values("Cost").iloc[0]

            st.success("Optimal Policy Found")

            st.metric("Reorder Point", int(best["RP"]))
            st.metric("Order Quantity", int(best["OQ"]))
            st.metric("Cost", round(best["Cost"],0))

        else:
            st.error("No feasible solution found")

        st.plotly_chart(px.scatter(results_df, x="Service Level", y="Cost"))
