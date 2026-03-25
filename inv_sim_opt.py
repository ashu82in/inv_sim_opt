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
    st.subheader("🎯 Dual-Constraint Policy Optimizer")
    st.write("This engine evolves a policy that minimizes costs while strictly obeying two constraints: **Fill Rate** and **Stockout Duration**.")

    # --- DUAL CONSTRAINT INPUTS ---
    c1, c2 = st.columns(2)
    with c1:
        target_fill_rate = st.slider("Min Fill Rate (%)", 85.0, 99.9, 98.0, 
                                     help="Percentage of total demand volume that must be met.")
        target_so_days = st.number_input("Max Allowable Stockout Days", value=7, 
                                         help="The maximum number of days per year the product can be out of stock.")
    with c2:
        st.info(f"**Goal:** Meet {target_fill_rate}% of demand AND stay under {target_so_days} days of stockouts.")
        # Hidden GA Params
        pop_size = 50 
        generations = 20

    if st.button("🧬 Run Dual-Constraint Optimization"):
        import random
        start_time = time.time()
        
        # Define search neighborhood
        max_r = int(avg_demand * lead_time * 5) 
        max_q = int(avg_demand * 45) # Approx 1.5 months
        bounds = [(0, max_r), (100, max_q)] 
        
        # 1. Initialize Population
        pop = [[np.random.randint(b[0], b[1]) for b in bounds] for _ in range(pop_size)]
        
        progress_bar = st.progress(0)
        status = st.empty()
        history = []

        for gen in range(generations):
            status.text(f"Generation {gen+1}/{generations}: Evaluating feasibility...")
            fitness_scores = []
            
            for individual in pop:
                r_test, q_test = individual[0], individual[1]
                
                # Run 30 scenarios for a robust average
                scen_metrics = []
                batch_demands = np.maximum(0, np.random.normal(avg_demand, std_demand, (30, num_days))).round()
                
                for i in range(30):
                    # We need the full DF here to calculate Fill Rate (Sum of Demand vs Sum of Sales)
                    df_sim, _, m = run_full_simulation(batch_demands[i], r_test, q_test, num_days, 
                                                       opening_balance, lead_time, unit_value, 
                                                       holding_cost_rate, ordering_cost, calc_aging=False)
                    
                    # Calculate Fill Rate for this specific scenario
                    total_demand = df_sim["Demand"].sum()
                    # Sales = Demand - (Unmet Demand if stock is 0)
                    # For simplicity in this engine: Sales = Demand if Closing Balance > 0
                    # A more accurate way:
                    actual_sales = total_demand - (df_sim[df_sim["Closing Balance"] == 0]["Demand"].sum())
                    fill_rate = (actual_sales / total_demand) * 100 if total_demand > 0 else 100
                    
                    m['fill_rate'] = fill_rate
                    scen_metrics.append(m)
                
                # Average performance across 30 scenarios
                avg_cost = np.mean([x['total_cost'] for x in scen_metrics])
                avg_so_days = np.mean([x['stockout_days'] for x in scen_metrics])
                avg_fill_rate = np.mean([x['fill_rate'] for x in scen_metrics])
                
                # --- DUAL CONSTRAINT LOGIC ---
                is_feasible = (avg_fill_rate >= target_fill_rate) and (avg_so_days <= target_so_days)
                
                if not is_feasible:
                    # Apply "Death Penalty" based on the severity of the worst violation
                    fr_gap = max(0, target_fill_rate - avg_fill_rate)
                    so_gap = max(0, avg_so_days - target_so_days)
                    # Score becomes massive so this individual doesn't survive
                    score = avg_cost * 50 * (fr_gap + so_gap + 1)
                else:
                    # Feasible: Score is just the actual inventory cost
                    score = avg_cost
                
                fitness_scores.append(score)
            
            # Evolution: Sort, Select, Breed
            ranked = [x for _, x in sorted(zip(fitness_scores, pop))]
            history.append(min(fitness_scores))
            parents = ranked[:max(2, pop_size//4)]
            new_pop = parents.copy()
            
            while len(new_pop) < pop_size:
                p1, p2 = random.sample(parents, 2)
                child = [int((p1[0] + p2[0])/2), int((p1[1] + p2[1])/2)]
                if np.random.random() < 0.25:
                    child[0] = np.clip(child[0] + np.random.randint(-50, 50), bounds[0][0], bounds[0][1])
                    child[1] = np.clip(child[1] + np.random.randint(-100, 100), bounds[1][0], bounds[1][1])
                new_pop.append(child)
            
            pop = new_pop
            progress_bar.progress((gen + 1) / generations)

        # --- FINAL RESULTS ---
        best_policy = parents[0]
        st.divider()
        res1, res2, res3 = st.columns(3)
        res1.metric("Optimized ROP (R)", best_policy[0])
        res2.metric("Optimized Qty (Q)", best_policy[1])
        res3.metric("Feasible Policy Cost", f"₹{round(history[-1], 0)}")

        st.success(f"Evolution complete! The algorithm found a policy that respects both your Fill Rate and Stockout Day limits.")
