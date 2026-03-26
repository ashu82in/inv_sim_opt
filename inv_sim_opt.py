import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import random
import scipy.stats as stats
from scipy.stats import norm

# --- INITIALIZE GLOBAL STATE ---
if 'req_fill_rate' not in st.session_state:
    st.session_state.req_fill_rate = 95.0
if 'req_so_days' not in st.session_state:
    st.session_state.req_so_days = 2
if 'best_policy' not in st.session_state:
    st.session_state.best_policy = None

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
    st.subheader("🎲 Monte Carlo Risk & Sensitivity Analysis")
    
    n_scenarios = st.number_input("Number of Scenarios to Simulate", min_value=1, max_value=100000, value=100)
    
    if st.button("🚀 Run Comprehensive Risk Test"):
        start_time = time.time()
        sensitivity_results = []
        
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        lt_tests = [lead_time, lead_time + 1, lead_time + 2]
        total_steps = len(lt_tests)
        
        for idx, lt_val in enumerate(lt_tests):
            status_text.text(f"Simulating Lead Time: {lt_val} days...")
            all_demands = np.maximum(0, np.random.normal(avg_demand, std_demand, (n_scenarios, num_days))).round()
            
            for i in range(n_scenarios):
                df_sim, _, metrics = run_full_simulation(
                    all_demands[i], reorder_point, order_qty, num_days, 
                    opening_balance, lt_val, unit_value, 
                    holding_cost_rate, ordering_cost, calc_aging=False
                )
                
                total_demand = all_demands[i].sum()
                unmet_demand = df_sim[df_sim["Closing Balance"] == 0]["Demand"].sum()
                metrics["fill_rate"] = ((total_demand - unmet_demand) / total_demand * 100) if total_demand > 0 else 100
                metrics["Tested LT"] = lt_val
                sensitivity_results.append(metrics)
            
            progress_bar.progress((idx + 1) / total_steps)
        
        res_df = pd.DataFrame(sensitivity_results)
        curr_df = res_df[res_df["Tested LT"] == lead_time]
        
        st.success(f"Simulated {len(res_df):,} total paths in {round(time.time()-start_time, 2)}s.")

        # --- SECTION 1: CORE KPIs ---
        st.write("### 📊 Service Level & Financial Summary")
        k1, k2, k3, k4 = st.columns(4)
        avg_fr = curr_df['fill_rate'].mean()
        k1.metric("Avg Fill Rate", f"{round(avg_fr, 2)}%")
        k2.metric("Avg Stockout Days", f"{round(curr_df['stockout_days'].mean(), 1)}")
        k3.metric("Avg Annual Cost", f"₹{round(curr_df['total_cost'].mean(), 0)}")
        k4.metric("WC Risk (95th Pctl)", f"₹{round(curr_df['avg_wc'].quantile(0.95), 0)}")

        # --- SECTION 2: STOCKOUT PROBABILITY (FIXED) ---
        st.write("#### 🛡️ Stockout Risk Probabilities")
        d_1 = round(num_days * 0.01, 1)
        d_5 = round(num_days * 0.05, 1)
        d_10 = round(num_days * 0.10, 1)

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            val1 = (curr_df['stockout_days'] == 0).sum() / n_scenarios * 100
            st.metric("Prob: No Stockouts", f"{round(val1, 2)}%")
            st.caption("Chance of zero stockout days.")
        with p2:
            val2 = (curr_df['stockout_days'] < (num_days * 0.01)).sum() / n_scenarios * 100
            st.metric("Stockouts < 1% Days", f"{round(val2, 2)}%")
            st.caption(f"Less Than {d_1} days in {round(val2, 1)}% of cases.")
        with p3:
            val3 = (curr_df['stockout_days'] < (num_days * 0.05)).sum() / n_scenarios * 100
            st.metric("Stockouts < 5% Days", f"{round(val3, 2)}%")
            st.caption(f"Less Than {d_5} days in {round(val3, 1)}% of cases.")
        with p4:
            val4 = (curr_df['stockout_days'] < (num_days * 0.10)).sum() / n_scenarios * 100
            st.metric("Stockouts < 10% Days", f"{round(val4, 2)}%")
            st.caption(f"Less Than {d_10} days in {round(val4, 1)}% of cases.")

        # --- SECTION 3: SENSITIVITY TABLE (Color Coded) ---
        st.divider()
        st.write("### 📋 Lead Time Sensitivity Table")
        sens_table = res_df.groupby("Tested LT").agg({
            "fill_rate": "mean",
            "stockout_days": "mean",
            "total_cost": "mean",
            "avg_wc": "mean"
        })
        sens_table.columns = ["Avg Fill Rate (%)", "Avg Stockout Days", "Avg Cost (₹)", "Avg Working Capital (₹)"]
        
        color_bad = 'background-color: #FF4B4B; color: black; font-weight: bold'
        color_good = 'background-color: #228B22; color: white; font-weight: bold'
        
        styled_sens = sens_table.style.format("{:.2f}")\
            .highlight_max(subset=["Avg Stockout Days", "Avg Cost (₹)", "Avg Working Capital (₹)"], props=color_bad)\
            .highlight_min(subset=["Avg Stockout Days", "Avg Cost (₹)", "Avg Working Capital (₹)"], props=color_good)\
            .highlight_max(subset=["Avg Fill Rate (%)"], props=color_good)\
            .highlight_min(subset=["Avg Fill Rate (%)"], props=color_bad)
        st.table(styled_sens)

        # --- SECTION 4: ALL 4 RISK DISTRIBUTIONS (Restored) ---
        st.divider()
        st.write("### 📈 Risk Distributions")
        
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            # Fill Rate Distribution
            fig_fr = px.histogram(curr_df, x="fill_rate", title="Fill Rate Distribution (Service Level)", color_discrete_sequence=['#00CC96'])
            st.plotly_chart(fig_fr, use_container_width=True)
        with row1_col2:
            # Cost Distribution
            fig_cost = px.histogram(curr_df, x="total_cost", title="Total Inventory Cost Distribution", color_discrete_sequence=['#EF553B'])
            st.plotly_chart(fig_cost, use_container_width=True)

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            # Stockout Severity
            fig_so = px.histogram(curr_df, x="stockout_days", title="Stockout Severity Distribution", color_discrete_sequence=['#FFA15A'])
            st.plotly_chart(fig_so, use_container_width=True)
        with row2_col2:
            # Avg Inventory
            fig_inv = px.histogram(curr_df, x="avg_inv", title="Average Inventory Level Distribution", color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig_inv, use_container_width=True)

        with st.expander("View Raw Statistical Data"):
            st.dataframe(curr_df.describe().T)


with tab3:
    st.header("🧬 AI Inventory Optimizer")
    st.write("Finds the lowest-cost ROP and Order Quantity (Q) that satisfies your service targets.")

    # --- 1. SET TARGETS & INPUTS ---
    c_input1, c_input2 = st.columns(2)
    with c_input1:
        # These are the variables that caused the NameError; defining them here fixes it.
        target_fr = st.slider("Target Annual Fill Rate (%)", 85.0, 100.0, 95.0, help="Percentage of total demand units fulfilled from stock.")
        target_so_days = st.number_input("Max Allowed Stockout Days", value=2, help="The 99% 'Worst Case' limit for days out of stock per year.")
    
    with c_input2:
        num_pop = 20  # Population of policies to test
        num_gen = 8   # Generations of evolution
        num_sim = 2000 # Scenarios per test (Vectorized makes this fast!)

    # --- 2. CALCULATE STATISTICAL BOUNDS ---
    import scipy.stats as stats
    # Setting a 'Lean' Floor at Avg Lead Time Demand to avoid overstocking
    avg_ltd = avg_demand * lead_time
    sigma_ltd = std_demand * np.sqrt(lead_time)
    
    rop_floor = int(max(0, avg_ltd)) 
    rop_ceil = int(avg_ltd + (5.5 * sigma_ltd)) # 5.5 Sigma Ceiling for extreme safety

    st.caption(f"Search Range: ROP between **{rop_floor}** and **{rop_ceil}** units.")

    # --- 3. OPTIMIZATION ENGINE ---
    if st.button("🚀 Run Vectorized Optimization"):
        start_time = time.time()
        
        # Pre-generate Demand Matrix (Scenarios x Days)
        # This is the secret to speed: Generate once, use many times.
        demand_matrix = np.maximum(0, np.random.normal(avg_demand, std_demand, (num_sim, num_days))).round()
        
        # Initialize Population [ROP, Q]
        bounds = [(rop_floor, rop_ceil), (100, int(avg_demand * 45))]
        pop = [[np.random.randint(b[0], b[1]) for b in bounds] for _ in range(num_pop)]
        
        progress_bar = st.progress(0)
        status_box = st.empty()
        history = []

        for gen in range(num_gen):
            fitness_scores = []
            
            # VECTORIZED EVALUATION OF POPULATION
            for r_t, q_t in pop:
                # Initialize simulation arrays
                inventory = np.full(num_sim, opening_balance, dtype=float)
                arrival_days = np.full(num_sim, -1)
                so_days = np.zeros(num_sim)
                total_unmet = np.zeros(num_sim)
                h_costs = np.zeros(num_sim)
                orders = np.zeros(num_sim)

                # Daily Loop (Vectorized across all scenarios)
                for d in range(num_days):
                    # Arrivals
                    arrived = (arrival_days == d)
                    inventory[arrived] += q_t
                    arrival_days[arrived] = -1
                    
                    # Demand & Shortfall
                    d_today = demand_matrix[:, d]
                    shortfall = np.maximum(0, d_today - inventory)
                    so_days[shortfall > 0] += 1
                    total_unmet += shortfall
                    
                    inventory = np.maximum(0, inventory - d_today)
                    h_costs += (inventory * (unit_value * holding_cost_rate / 365))
                    
                    # Reordering
                    reorder_mask = (inventory <= r_t) & (arrival_days == -1)
                    arrival_days[reorder_mask] = d + lead_time
                    orders[reorder_mask] += 1

                # Calculate KPIs
                total_demand_per_scenario = demand_matrix.sum(axis=1)
                fill_rates = (1 - (total_unmet / total_demand_per_scenario)) * 100
                scenario_costs = h_costs + (orders * ordering_cost)
                
                # Percentile Risk Checks
                p99_so = np.percentile(so_days, 99)
                p1_fr = np.percentile(fill_rates, 1)
                avg_cost = np.mean(scenario_costs)

                # Fitness: Cost + Massive Penalty for failing constraints
                if p99_so <= target_so_days and p1_fr >= target_fr:
                    score = avg_cost
                else:
                    # Penalty scales with how far off they are
                    penalty = 1 + (p99_so - target_so_days) + (target_fr - p1_fr)
                    score = avg_cost * 1000 * penalty
                
                fitness_scores.append(score)

            # --- EVOLUTION ---
            ranked = [x for _, x in sorted(zip(fitness_scores, pop))]
            best_score = min(fitness_scores)
            
            # Only record history if we found a feasible solution
            history.append(best_score if best_score < 1e7 else None)
            
            # Elitism + Crossover
            new_pop = ranked[:2]
            while len(new_pop) < num_pop:
                p1, p2 = random.sample(ranked[:8], 2)
                child = [int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)]
                if np.random.random() < 0.2: # Mutation
                    child[0] = np.clip(child[0] + np.random.randint(-30, 30), rop_floor, rop_ceil)
                new_pop.append(child)
            pop = new_pop
            
            progress_bar.progress((gen + 1) / num_gen)
            status_box.write(f"🧬 **Generation {gen+1}**: " + 
                             (f"Best Cost ₹{round(best_score,0)}" if best_score < 1e7 else "Searching for safe zone..."))

        # --- FINAL RESULTS ---
        best_r, best_q = ranked[0]
        st.divider()
        st.success(f"Optimization Complete in {round(time.time()-start_time, 2)}s")
        
        res1, res2, res3 = st.columns(3)
        res1.metric("Recommended ROP", best_r)
        res2.metric("Recommended Qty", best_q)
        res3.metric("Annualized Cost", f"₹{round(best_score if best_score < 1e7 else 0, 0)}")

        # Convergence Plot
        plot_history = [h for h in history if h is not None]
        if plot_history:
            st.plotly_chart(px.line(y=plot_history, title="Minima Convergence (Cost Optimization Path)", 
                                   labels={'y':'Annual Cost ₹','x':'Generation'}, markers=True))
        else:
            st.error("⚠️ No feasible policy found. Try increasing 'Max Stockout Days' or lowering 'Fill Rate %'.")
