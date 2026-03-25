import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(layout="wide")
st.title("📦 Inventory Policy & Risk Simulator")

# =========================================================
# SIDEBAR INPUTS (Shared across tabs)
# =========================================================
st.sidebar.header("Global Inventory Inputs")
opening_balance = st.sidebar.number_input("Opening Balance", value=500)
avg_demand = st.sidebar.number_input("Average Daily Demand", value=25)
cov = st.sidebar.number_input("Coefficient of Variation", value=0.8)
lead_time = st.sidebar.number_input("Lead Time (Days)", value=3)
reorder_point_input = st.sidebar.number_input("Manual Reorder Point", value=200)
order_qty = st.sidebar.number_input("Order Quantity", value=300)
unit_value = st.sidebar.number_input("Value Per Unit", value=100)
holding_cost_percent = st.sidebar.number_input("Annual Holding Cost (%)", value=20.0)
ordering_cost = st.sidebar.number_input("Ordering Cost per Order", value=500)
num_days = st.sidebar.slider("Simulation Days", 100, 2000, 365)

std_demand = avg_demand * cov
holding_cost_rate = holding_cost_percent / 100

# =========================================================
# REUSABLE SIMULATION ENGINE
# =========================================================
def run_core_simulation(demand_seq, r_point, q_qty):
    inventory_layers = [{"qty": opening_balance, "age": 0}]
    pipeline_orders = []
    results = []
    
    for day in range(num_days):
        # Age and Receive
        for l in inventory_layers: l["age"] += 1
        for order in pipeline_orders.copy():
            if order[0] == day:
                inventory_layers.append({"qty": order[1], "age": 0})
                pipeline_orders.remove(order)

        # Consume Demand (FIFO)
        rem = demand_seq[day]
        while rem > 0 and inventory_layers:
            if inventory_layers[0]["qty"] <= rem:
                rem -= inventory_layers[0]["qty"]
                inventory_layers.pop(0)
            else:
                inventory_layers[0]["qty"] -= rem
                rem = 0
        
        curr_inv = sum(l["qty"] for l in inventory_layers)
        pipe_qty = sum(q for _, q in pipeline_orders)
        inv_pos = curr_inv + pipe_qty
        
        placed_order = 0
        if inv_pos < r_point:
            placed_order = q_qty
            pipeline_orders.append((day + lead_time, q_qty))
            
        results.append({
            "Closing": curr_inv,
            "Position": inv_pos,
            "Ordered": placed_order,
            "Demand": demand_seq[day]
        })
    
    res_df = pd.DataFrame(results)
    
    # Calculate Metrics for this specific run
    stockouts = (res_df["Closing"] == 0).sum()
    avg_wc = res_df["Position"].mean() * unit_value
    min_inv = res_df["Closing"].min()
    avg_age = (res_df["Position"].mean()) / (avg_demand if avg_demand > 0 else 1)
    
    total_hold = (res_df["Position"] * unit_value * holding_cost_rate / 365).sum()
    total_ord = (res_df["Ordered"] > 0).sum() * ordering_cost
    
    return {
        "Stockouts": stockouts,
        "AvgWC": avg_wc,
        "MinInv": min_inv,
        "AvgAge": avg_age,
        "TotalCost": total_hold + total_ord,
        "FullDF": res_df
    }

# =========================================================
# TABS
# =========================================================
tab1, tab2 = st.tabs(["📈 Single Scenario Analysis", "🎲 Monte Carlo Risk Simulation"])

with tab1:
    # Service Level Logic
    use_service_level = st.checkbox("Use Service Level for ROP", key="tab1_sl")
    if use_service_level:
        z = norm.ppf(st.slider("Target SL", 0.80, 0.99, 0.95))
        reorder_point = int(avg_demand*lead_time + z*std_demand*np.sqrt(lead_time))
        st.success(f"Calculated ROP: {reorder_point}")
    else:
        reorder_point = reorder_point_input

    if st.button("Run Single Simulation"):
        # Generate one static demand sequence
        static_demand = np.maximum(0, np.random.normal(avg_demand, std_demand, num_days)).round()
        sim_data = run_core_simulation(static_demand, reorder_point, order_qty)
        
        # Display KPIs (Condensed for brevity, reuse your original metric code here)
        st.metric("Total Cost", f"₹{round(sim_data['TotalCost'],0)}")
        
        # Plotting (Reuse your original Plotly code here)
        fig = px.line(sim_data["FullDF"], y="Closing", title="Inventory Levels")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Monte Carlo Risk Assessment")
    st.write("Simulate hundreds of possible futures to see the distribution of outcomes.")
    
    num_scenarios = st.number_input("Number of Scenarios", min_value=10, max_value=1000, value=100)
    
    if st.button(f"Run {num_scenarios} Scenarios"):
        all_metrics = []
        
        progress_bar = st.progress(0)
        for i in range(num_scenarios):
            # Generate unique random demand for every scenario
            s_demand = np.maximum(0, np.random.normal(avg_demand, std_demand, num_days)).round()
            m = run_core_simulation(s_demand, reorder_point_input, order_qty)
            all_metrics.append(m)
            progress_bar.progress((i + 1) / num_scenarios)
        
        m_df = pd.DataFrame(all_metrics)

        # --- Visualizing Distributions ---
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Inventory Cost Distribution")
            st.plotly_chart(px.histogram(m_df, x="TotalCost", nbins=20, color_discrete_sequence=['indianred']), use_container_width=True)
            
            st.write("### Stockout Days Distribution")
            st.plotly_chart(px.histogram(m_df, x="Stockouts", nbins=15, color_discrete_sequence=['orange']), use_container_width=True)

        with col2:
            st.write("### Avg Working Capital Distribution")
            st.plotly_chart(px.histogram(m_df, x="AvgWC", nbins=20, color_discrete_sequence=['royalblue']), use_container_width=True)
            
            st.write("### Minimum Inventory Distribution")
            st.plotly_chart(px.histogram(m_df, x="MinInv", nbins=15, color_discrete_sequence=['green']), use_container_width=True)

        st.write("### Summary Statistics")
        st.table(m_df[["TotalCost", "AvgWC", "Stockouts", "AvgAge"]].describe().T)
