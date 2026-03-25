import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(layout="wide")
st.title("Inventory Policy & Risk Simulator")

# =========================================================
# SHARED SIMULATION ENGINE
# =========================================================
def run_full_simulation(demand_seq, r_point, q_qty, num_days_sim, open_bal, l_time, val_unit, h_rate, o_cost):
    inventory_layers = [{"qty": open_bal, "age": 0}]
    pipeline_orders = []
    data = []
    aging_data = []

    for day in range(num_days_sim):
        # Age inventory
        for layer in inventory_layers:
            layer["age"] += 1

        # Receive orders
        for order in pipeline_orders.copy():
            if order[0] == day:
                inventory_layers.append({"qty": order[1], "age": 0})
                pipeline_orders.remove(order)

        # Demand (FIFO)
        remaining = demand_seq[day]
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
        if inventory_position < r_point:
            new_order = q_qty
            pipeline_orders.append((day + l_time, q_qty))

        data.append([demand_seq[day], total_inventory, inventory_position, new_order])

        # Aging buckets
        bucket = {"0-30":0, "31-60":0, "61-90":0, "90+":0}
        for l in inventory_layers:
            age, qty = l["age"], l["qty"]
            if age <= 30: bucket["0-30"] += qty
            elif age <= 60: bucket["31-60"] += qty
            elif age <= 90: bucket["61-90"] += qty
            else: bucket["90+"] += qty
        bucket["Day"] = day
        aging_data.append(bucket)

    df = pd.DataFrame(data, columns=["Demand","Closing Balance","Inventory Position","New Order"])
    aging_df = pd.DataFrame(aging_data)
    
    # Calculate Summary Metrics for the Engine
    df["Inventory Value"] = df["Inventory Position"] * val_unit
    df["Holding Cost"] = df["Inventory Value"] * h_rate / 365
    
    metrics = {
        "stockout_days": (df["Closing Balance"] == 0).sum(),
        "avg_inv": df["Inventory Position"].mean(),
        "avg_wc": df["Inventory Position"].mean() * val_unit,
        "min_inv": df["Closing Balance"].min(),
        "max_inv": df["Closing Balance"].max(),
        "total_cost": df["Holding Cost"].sum() + ((df["New Order"] > 0).sum() * o_cost),
        "avg_age": df["Inventory Position"].mean() / (demand_seq.mean() if demand_seq.mean() > 0 else 1)
    }
    
    return df, aging_df, metrics

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

std_demand = avg_demand * cov
holding_cost_rate = holding_cost_percent / 100

# =========================================================
# TABS SETUP
# =========================================================
tab1, tab2 = st.tabs(["📊 Single Scenario (Detailed)", "🎲 Monte Carlo Simulation"])

with tab1:
    # --- Policy Input ---
    use_service_level = st.checkbox("Use Service Level", key="use_sl")
    service_level_input = st.slider("Target Service Level", 0.80, 0.99, 0.95, disabled=not use_service_level)
    
    reorder_point = reorder_point_input
    if use_service_level:
        z = norm.ppf(service_level_input)
        reorder_point = int(avg_demand*lead_time + z*std_demand*np.sqrt(lead_time))
        st.success(f"Auto Reorder Point: {reorder_point}")

    if st.button("Reset & Run Single Scenario"):
        st.session_state.demand_sequence = np.maximum(0, np.random.normal(avg_demand, std_demand, num_days)).round()

    if "demand_sequence" not in st.session_state:
        st.session_state.demand_sequence = np.maximum(0, np.random.normal(avg_demand, std_demand, num_days)).round()

    # Run Simulation
    df, aging_df, m = run_full_simulation(st.session_state.demand_sequence, reorder_point, order_qty, num_days, opening_balance, lead_time, unit_value, holding_cost_rate, ordering_cost)
    df["Date"] = pd.date_range("2024-01-01", periods=num_days)
    aging_df["Date"] = df["Date"]

    # --- KPI Dashboards (Restored exactly as per your code) ---
    st.subheader("Inventory KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stockout Days", m['stockout_days'])
    c2.metric("Average Age of Inventory", round(m['avg_age'], 1))
    c3.metric("Average Inventory", round(m['avg_inv'], 0))
    c4.metric("Avg Working Capital", round(m['avg_wc'], 0))

    st.subheader("Inventory Range")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Minimum Inventory", round(m['min_inv'], 0))
    r2.metric("Maximum Inventory", round(m['max_inv'], 0))
    r3.metric("Min Working Capital", round(m['min_inv'] * unit_value, 0))
    r4.metric("Max Working Capital", round(m['max_inv'] * unit_value, 0))

    # --- Charts (Restored) ---
    st.subheader("Inventory Behaviour")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance"], name="Inventory"))
    fig.add_hline(y=reorder_point, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Aging Buckets")
    for col in ["0-30","31-60","61-90","90+"]: aging_df[col] *= unit_value
    aging_melt = aging_df.melt(id_vars=["Date"], value_vars=["0-30","31-60","61-90","90+"], var_name="Bucket", value_name="Value")
    fig_bucket = px.bar(aging_melt, x="Date", y="Value", color="Bucket")
    st.plotly_chart(fig_bucket, use_container_width=True)

with tab2:
    st.subheader("Multi-Scenario Simulation")
    n_scenarios = st.number_input("Number of Scenarios", 10, 500, 100)
    
    if st.button("Run Monte Carlo"):
        all_results = []
        progress_bar = st.progress(0)
        
        for i in range(n_scenarios):
            # Generate fresh random demand for each scenario
            scen_demand = np.maximum(0, np.random.normal(avg_demand, std_demand, num_days)).round()
            _, _, metrics = run_full_simulation(scen_demand, reorder_point, order_qty, num_days, opening_balance, lead_time, unit_value, holding_cost_rate, ordering_cost)
            all_results.append(metrics)
            progress_bar.progress((i + 1) / n_scenarios)
        
        results_df = pd.DataFrame(all_results)
        
        # --- Monte Carlo Visualizations ---
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Total Inventory Cost Distribution")
            st.plotly_chart(px.histogram(results_df, x="total_cost", nbins=20, color_discrete_sequence=['indianred']), use_container_width=True)
            
            st.write("### Average Working Capital Distribution")
            st.plotly_chart(px.histogram(results_df, x="avg_wc", nbins=20, color_discrete_sequence=['royalblue']), use_container_width=True)

        with col2:
            st.write("### Stockout Days Distribution")
            st.plotly_chart(px.histogram(results_df, x="stockout_days", nbins=15, color_discrete_sequence=['orange']), use_container_width=True)

            st.write("### Minimum Inventory Distribution")
            st.plotly_chart(px.histogram(results_df, x="min_inv", nbins=15, color_discrete_sequence=['green']), use_container_width=True)

        st.write("### Risk Summary Statistics")
        st.dataframe(results_df.describe().T)
