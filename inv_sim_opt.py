import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import random
from scipy.stats import norm


st.set_page_config(layout="wide")
st.title("📦 Inventory Policy & Risk Simulator")

# =========================================================
# SHARED SIMULATION ENGINE
# =========================================================
# =========================================================
# SHARED SIMULATION ENGINE (FIXED)
# =========================================================
def run_full_simulation(demand_seq, r_point, q_qty, num_days_sim, open_bal, l_time, val_unit, h_rate, o_cost, calc_aging=True):
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

        # ONLY calculate aging if calc_aging is True (saves massive time in Tab 2)
        if calc_aging:
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
    aging_df = pd.DataFrame(aging_data) if calc_aging else pd.DataFrame()
    
    # Core Metrics for Engine
    df["Blocked Working Capital"] = df["Inventory Position"] * val_unit
    df["Holding Cost"] = df["Blocked Working Capital"] * h_rate / 365
    
    metrics = {
        "stockout_days": (df["Closing Balance"] == 0).sum(),
        "avg_inv": df["Inventory Position"].mean(),
        "avg_wc": df["Blocked Working Capital"].mean(),
        "min_inv": df["Closing Balance"].min(),
        "max_inv": df["Closing Balance"].max(),
        "min_wc": df["Blocked Working Capital"].min(),
        "max_wc": df["Blocked Working Capital"].max(),
        "total_holding": df["Holding Cost"].sum(),
        "num_orders": (df["New Order"] > 0).sum(),
        "avg_age": df["Inventory Position"].mean() / (demand_seq.mean() if demand_seq.mean() > 0 else 1)
    }
    metrics["total_cost"] = metrics["total_holding"] + (metrics["num_orders"] * o_cost)
    
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
tab1, tab2, tab3 = st.tabs(["📊 Detailed Analysis", "🎲 Monte Carlo Simulation", "🎯 Policy Optimizer"])

with tab1:
    # --- Policy Input ---
    st.subheader("Policy Input")
    use_service_level = st.checkbox("Use Service Level", key="use_sl")
    service_level_input = st.slider("Target Service Level", 0.80, 0.99, 0.95, disabled=not use_service_level)
    
    reorder_point = reorder_point_input
    if use_service_level:
        z = norm.ppf(service_level_input)
        reorder_point = int(avg_demand*lead_time + z*std_demand*np.sqrt(lead_time))
        st.success(f"Auto Reorder Point: {reorder_point}")

    if st.button("Reset Demand Scenario"):
        st.session_state.demand_sequence = np.maximum(0, np.random.normal(avg_demand, std_demand, num_days)).round()

    if "demand_sequence" not in st.session_state:
        st.session_state.demand_sequence = np.maximum(0, np.random.normal(avg_demand, std_demand, num_days)).round()

    # Run Simulation
    df, aging_df, m = run_full_simulation(st.session_state.demand_sequence, reorder_point, order_qty, num_days, opening_balance, lead_time, unit_value, holding_cost_rate, ordering_cost)
    df["Date"] = pd.date_range("2024-01-01", periods=num_days)
    aging_df["Date"] = df["Date"]

    # --- KPI DISPLAY (Restored) ---
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
    r3.metric("Minimum Working Capital", round(m['min_wc'], 0))
    r4.metric("Maximum Working Capital", round(m['max_wc'], 0))

    st.subheader("Inventory Cost Metrics")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Total Holding Cost", round(m['total_holding'], 0))
    cc2.metric("Total Ordering Cost", round(m['num_orders'] * ordering_cost, 0))
    cc3.metric("Total Inventory Cost", round(m['total_cost'], 0))

    # --- EOQ LOGIC (Restored) ---
    st.subheader("EOQ")
    annual_demand = avg_demand * 365
    holding_cost_per_unit = unit_value * holding_cost_rate
    eoq = int(np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit))
    
    # Quick Simulation for EOQ Comparison
    _, _, m_eoq = run_full_simulation(st.session_state.demand_sequence, reorder_point, eoq, num_days, opening_balance, lead_time, unit_value, holding_cost_rate, ordering_cost)
    
    e1, e2 = st.columns(2)
    e1.metric("Economic Order Quantity", eoq)
    e2.metric("Selected Order Quantity", order_qty)

    st.subheader("Cost Comparison")
    k1, k2, k3 = st.columns(3)
    k1.metric("Cost with Current Policy", round(m['total_cost'], 0))
    k2.metric("Cost with EOQ", round(m_eoq['total_cost'], 0))
    k3.metric("Savings Using EOQ", round(m['total_cost'] - m_eoq['total_cost'], 0))

    # --- Charts (Restored) ---
    st.subheader("Inventory Behaviour")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance"], name="Inventory"))
    fig.add_hline(y=reorder_point, line_dash="dash")
    fig.add_hrect(y0=0, y1=reorder_point*0.5, fillcolor="red", opacity=0.1)
    fig.add_hrect(y0=reorder_point*0.5, y1=reorder_point, fillcolor="yellow", opacity=0.1)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Blocked Working Capital")
    fig_wc = px.line(df, x="Date", y="Blocked Working Capital")
    fig_wc.add_hline(y=m['avg_wc'], line_dash="dash")
    st.plotly_chart(fig_wc, use_container_width=True)

    st.subheader("Aging Buckets")
    temp_aging = aging_df.copy()
    for col in ["0-30","31-60","61-90","90+"]: temp_aging[col] *= unit_value
    aging_melt = temp_aging.melt(id_vars=["Date"], value_vars=["0-30","31-60","61-90","90+"], var_name="Bucket", value_name="Value")
    fig_bucket = px.bar(aging_melt, x="Date", y="Value", color="Bucket", color_discrete_map={"0-30":"#ADD8E6","31-60":"#6495ED","61-90":"#FFA07A","90+":"#FF0000"})
    st.plotly_chart(fig_bucket, use_container_width=True)

    st.subheader("Demand Distribution")
    st.plotly_chart(px.histogram(df, x="Demand"), use_container_width=True)

    # --- Data Table Toggle (Restored) ---
    if st.checkbox("Show Simulation Data"):
        st.subheader("Simulation Data")
        st.dataframe(df)

with tab2:
    st.subheader("Inventory Simulation")
    
    # Input for high-volume scenarios (Supports up to 100,000)
    n_scenarios = st.number_input("Number of Scenarios to Simulate", min_value=1, max_value=100000, value=100)
    
    if st.button("🚀 Run Comprehensive Risk Test"):
        start_time = time.time()
        sensitivity_results = []
        
        # UI Feedback for long runs
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        # Test Current Lead Time, +1, and +2 for Sensitivity
        lt_tests = [lead_time, lead_time + 1, lead_time + 2]
        total_steps = len(lt_tests)
        
        for idx, lt_val in enumerate(lt_tests):
            status_text.text(f"Simulating Lead Time: {lt_val} days...")
            
            # Vectorized demand generation for speed
            all_demands = np.maximum(0, np.random.normal(avg_demand, std_demand, (n_scenarios, num_days))).round()
            
            for i in range(n_scenarios):
                # calc_aging=False is used here to keep memory usage low for 100k+ runs
                _, _, metrics = run_full_simulation(
                    all_demands[i], reorder_point, order_qty, num_days, 
                    opening_balance, lt_val, unit_value, 
                    holding_cost_rate, ordering_cost, calc_aging=False
                )
                metrics["Tested LT"] = lt_val
                sensitivity_results.append(metrics)
            
            progress_bar.progress((idx + 1) / total_steps)
        
        status_text.text("Processing Results...")
        res_df = pd.DataFrame(sensitivity_results)
        curr_df = res_df[res_df["Tested LT"] == lead_time]
        
        elapsed = time.time() - start_time
        st.success(f"Successfully simulated {len(res_df):,} total paths in {round(elapsed, 2)} seconds.")

        # --- Probability KPIs (Refined Concise Phrasing) ---
        st.write("### 📊 Probability & Risk Metrics (Current Policy)")
        
        # Calculate thresholds based on the actual simulation length
        d_1 = round(num_days * 0.01, 1)
        d_5 = round(num_days * 0.05, 1)
        d_10 = round(num_days * 0.10, 1)

        k1, k2, k3, k4 = st.columns(4)
        
        with k1:
            p_no_so = (curr_df['stockout_days'] == 0).sum() / n_scenarios * 100
            st.metric("Prob: No Stockouts", f"{round(p_no_so, 2)}%")
            st.caption("Chance of zero stockout days.")

        with k2:
            p_1 = (curr_df['stockout_days'] < (num_days * 0.01)).sum() / n_scenarios * 100
            st.metric("Stockouts < 1% Days", f"{round(p_1, 2)}%")
            st.caption(f"Less Than {d_1} days in {round(p_1, 1)}% of cases.")

        with k3:
            p_5 = (curr_df['stockout_days'] < (num_days * 0.05)).sum() / n_scenarios * 100
            st.metric("Stockouts < 5% Days", f"{round(p_5, 2)}%")
            st.caption(f"Less Than {d_5} days in {round(p_5, 1)}% of cases.")

        with k4:
            p_10 = (curr_df['stockout_days'] < (num_days * 0.10)).sum() / n_scenarios * 100
            st.metric("Stockouts < 10% Days", f"{round(p_10, 2)}%")
            st.caption(f"Less Than {d_10} days in {round(p_10, 1)}% of cases.")

        st.divider()

       # --- Lead Time Sensitivity Table with Bold Color Coding ---
        st.write("### 📋 Lead Time Sensitivity Table")
        
        # Define the summary table
        sens_table = res_df.groupby("Tested LT").agg({
            "stockout_days": ["mean", lambda x: (x == 0).sum() / n_scenarios * 100],
            "total_cost": "mean",
            "avg_wc": "mean"
        })
        sens_table.columns = ["Avg Stockout Days", "Prob: No Stockout (%)", "Avg Cost (₹)", "Avg Working Capital (₹)"]
        
        # Define Bold Hex Colors
        color_bad = 'background-color: #FF4B4B; color: black; font-weight: bold' # Bold Red
        color_good = 'background-color: #228B22; color: white; font-weight: bold' # Bold Green (White text for contrast)
        
        # Apply Logic-Based Highlighting
        styled_table = sens_table.style.format("{:.2f}")\
            .highlight_max(subset=["Avg Stockout Days", "Avg Cost (₹)", "Avg Working Capital (₹)"], props=color_bad)\
            .highlight_min(subset=["Avg Stockout Days", "Avg Cost (₹)", "Avg Working Capital (₹)"], props=color_good)\
            .highlight_max(subset=["Prob: No Stockout (%)"], props=color_good)\
            .highlight_min(subset=["Prob: No Stockout (%)"], props=color_bad)

        st.table(styled_table)

        # =========================================================
        # 📈 RISK DISTRIBUTIONS (ALL 4 CHARTS)
        # =========================================================
        st.divider()
        st.write("### 📈 Risk Distributions (Current Lead Time)")
        
        # Row 1: Costs and Stockouts
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            fig_cost = px.histogram(curr_df, x="total_cost", title="Annual Inventory Cost Frequency", color_discrete_sequence=['#EF553B'])
            fig_cost.add_vline(x=curr_df['total_cost'].mean(), line_dash="dash", line_color="black", annotation_text="Mean")
            st.plotly_chart(fig_cost, use_container_width=True)
        with r1c2:
            st.plotly_chart(px.histogram(curr_df, x="stockout_days", title="Stockout Severity Distribution", color_discrete_sequence=['#FFA15A']), use_container_width=True)

        # Row 2: Inventory Levels
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.plotly_chart(px.histogram(curr_df, x="avg_inv", title="Average Inventory Distribution", color_discrete_sequence=['#636EFA']), use_container_width=True)
        with r2c2:
            st.plotly_chart(px.histogram(curr_df, x="min_inv", title="Minimum Inventory (Safety Buffer) Distribution", color_discrete_sequence=['#00CC96']), use_container_width=True)

        # Statistical Summary Table
        with st.expander("View Full Statistical Raw Data"):
            st.dataframe(curr_df.describe().T)


# Add "Optimization" to the tabs setup line: 
# tab1, tab2, tab3 = st.tabs(["📊 Detailed Analysis", "🎲 Monte Carlo Simulation", "🎯 Policy Optimizer"])

with tab3:
    st.subheader("🧬 Genetic Algorithm Policy Optimizer")
    st.write("Using metaheuristics to evolve the most cost-effective inventory policy for volatile demand.")

    col1, col2 = st.columns(2)
    with col1:
        pop_size = st.select_slider("Population Size", options=[10, 20, 50], value=20)
        generations = st.select_slider("Number of Generations", options=[5, 10, 20], value=10)
    with col2:
        stockout_penalty = st.number_input("Stockout Penalty (₹/unit)", value=unit_value*2.0)
        st.info("Higher penalties force the model to prioritize Safety Stock.")

    if st.button("🧬 Evolve Best Policy"):
        # --- GA SETTINGS ---
        # Gene 0: ROP, Gene 1: Qty
        bounds = [(0, 1000), (100, 2000)] 
        
        # 1. Initialize Population
        pop = []
        for _ in range(pop_size):
            pop.append([np.random.randint(bounds[0][0], bounds[0][1]), 
                        np.random.randint(bounds[1][0], bounds[1][1])])
        
        progress_bar = st.progress(0)
        status = st.empty()
        history = []

        for gen in range(generations):
            status.text(f"Generation {gen+1}/{generations}: Evolving policies...")
            scores = []
            
            # 2. Fitness Evaluation (Simulation)
            for individual in pop:
                r_t, q_t = individual[0], individual[1]
                # Run 30 scenarios per individual to average out noise
                sim_costs = []
                for _ in range(30):
                    s_dem = np.maximum(0, np.random.normal(avg_demand, std_demand, num_days)).round()
                    _, _, m = run_full_simulation(s_dem, r_t, q_t, num_days, opening_balance, lead_time, unit_value, holding_cost_rate, ordering_cost, calc_aging=False)
                    total_p_cost = m['total_cost'] + (m['stockout_days'] * avg_demand * stockout_penalty)
                    sim_costs.append(total_p_cost)
                
                scores.append(np.mean(sim_costs))
            
            # Rank Population (Lower cost is better)
            ranked = [x for _, x in sorted(zip(scores, pop))]
            best_score = min(scores)
            history.append(best_score)
            
            # 3. Selection (Keep top 25% as parents)
            parents = ranked[:max(2, pop_size//4)]
            
            # 4. Crossover & Mutation (Create next generation)
            new_pop = parents.copy() # Elitism: Keep best parents
            while len(new_pop) < pop_size:
                p1, p2 = random.sample(parents, 2)
                # Crossover
                child = [int((p1[0] + p2[0])/2), int((p1[1] + p2[1])/2)]
                # Mutation (15% chance)
                if np.random.random() < 0.15:
                    child[0] = np.clip(child[0] + np.random.randint(-50, 50), bounds[0][0], bounds[0][1])
                    child[1] = np.clip(child[1] + np.random.randint(-100, 100), bounds[1][0], bounds[1][1])
                new_pop.append(child)
            
            pop = new_pop
            progress_bar.progress((gen + 1) / generations)

        # Final Results
        best_r, best_q = parents[0][0], parents[0][1]
        
        st.divider()
        res1, res2, res3 = st.columns(3)
        res1.metric("Evolved ROP", best_r)
        res2.metric("Evolved Quantity", best_q)
        res3.metric("Min Avg Cost", f"₹{round(best_score, 0)}")

        # --- Convergence Chart ---
        st.write("### 📉 Convergence (Cost Reduction Over Generations)")
        fig_conv = px.line(x=range(1, generations+1), y=history, labels={'x':'Generation', 'y':'Total Cost'}, markers=True)
        st.plotly_chart(fig_conv, use_container_width=True)
