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

import numpy as np
import pandas as pd
import time
import plotly.express as px

with tab2:
    st.subheader("🎲 Monte Carlo Risk & Sensitivity Analysis")
    
    n_scenarios = st.number_input("Number of Scenarios to Simulate", min_value=1, max_value=100000, value=1000)
    
    if st.button("🚀 Run Comprehensive Vectorized Test"):
        start_time = time.time()
        
        # 1. SETUP SENSITIVITY TESTS
        lt_tests = [lead_time, lead_time + 1, lead_time + 2]
        sensitivity_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 2. RUN SIMULATION FOR EACH LEAD TIME
        for idx, lt_val in enumerate(lt_tests):
            status_text.text(f"Simulating Lead Time: {lt_val} days...")
            
            # Generate demand once for this LT test
            demand_matrix = np.maximum(0, np.random.normal(avg_demand, std_demand, (n_scenarios, num_days))).round()
            
            # Initialize Vectorized Arrays
            inventory = np.full(n_scenarios, opening_balance, dtype=float)
            arrival_days = np.full(n_scenarios, -1)
            so_days = np.zeros(n_scenarios)
            total_unmet = np.zeros(n_scenarios)
            h_costs_accum = np.zeros(n_scenarios)
            order_counts = np.zeros(n_scenarios)
            daily_inv_sum = np.zeros(n_scenarios)

            # Daily Simulation Loop (Vectorized across scenarios)
            for d in range(num_days):
                # Arrivals
                arrived = (arrival_days == d)
                inventory[arrived] += order_qty
                arrival_days[arrived] = -1
                
                # Demand Fulfillment
                d_today = demand_matrix[:, d]
                shortfall = np.maximum(0, d_today - inventory)
                so_days[shortfall > 0] += 1
                total_unmet += shortfall
                
                inventory = np.maximum(0, inventory - d_today)
                
                # Financials
                h_costs_accum += (inventory * (unit_value * holding_cost_rate / 365))
                daily_inv_sum += inventory
                
                # Reorder Logic
                reorder_mask = (inventory <= reorder_point) & (arrival_days == -1)
                arrival_days[reorder_mask] = d + lt_val
                order_counts[reorder_mask] += 1

            # 3. CONSTRUCT RESULTS FOR THIS LEAD TIME
            total_annual_demand = demand_matrix.sum(axis=1)
            fill_rates = (1 - (total_unmet / total_annual_demand)) * 100
            avg_inv_levels = daily_inv_sum / num_days
            avg_wc = avg_inv_levels * unit_value
            total_costs = h_costs_accum + (order_counts * ordering_cost)

            # Create a dataframe for this LT and append
            lt_df = pd.DataFrame({
                "fill_rate": fill_rates,
                "stockout_days": so_days,
                "total_cost": total_costs,
                "avg_inv": avg_inv_levels,
                "avg_wc": avg_wc,
                "Tested LT": lt_val
            })
            sensitivity_results.append(lt_df)
            progress_bar.progress((idx + 1) / len(lt_tests))

        # 4. DATA PROCESSING
        res_df = pd.concat(sensitivity_results)
        curr_df = res_df[res_df["Tested LT"] == lead_time]
        st.success(f"Simulated {len(res_df):,} total paths in {round(time.time()-start_time, 2)}s.")

        # --- SECTION 1: CORE KPIs ---
        st.write("### 📊 Service Level & Financial Summary")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Avg Fill Rate", f"{round(curr_df['fill_rate'].mean(), 2)}%")
        k2.metric("Avg Stockout Days", f"{round(curr_df['stockout_days'].mean(), 1)}")
        k3.metric("Avg Annual Cost", f"₹{round(curr_df['total_cost'].mean(), 0)}")
        k4.metric("WC Risk (95th Pctl)", f"₹{round(curr_df['avg_wc'].quantile(0.95), 0)}")

        # --- SECTION 2: STOCKOUT PROBABILITY ---
        st.write("#### 🛡️ Stockout Risk Probabilities")
        p1, p2, p3, p4 = st.columns(4)
        # Probabilities calculated directly from vectorized results
        p1.metric("Prob: No Stockouts", f"{round((curr_df['stockout_days'] == 0).mean() * 100, 2)}%")
        p2.metric("Stockouts < 1% Days", f"{round((curr_df['stockout_days'] < num_days*0.01).mean() * 100, 2)}%")
        p3.metric("Stockouts < 5% Days", f"{round((curr_df['stockout_days'] < num_days*0.05).mean() * 100, 2)}%")
        p4.metric("Stockouts < 10% Days", f"{round((curr_df['stockout_days'] < num_days*0.10).mean() * 100, 2)}%")

        # --- SECTION 3: SENSITIVITY TABLE ---
        st.divider()
        st.write("### 📋 Lead Time Sensitivity Table")
        sens_table = res_df.groupby("Tested LT").agg({
            "fill_rate": "mean", "stockout_days": "mean", "total_cost": "mean", "avg_wc": "mean"
        })
        sens_table.columns = ["Avg Fill Rate (%)", "Avg Stockout Days", "Avg Cost (₹)", "Avg Working Capital (₹)"]
        
        # Styling (Same as your original code)
        styled_sens = sens_table.style.format("{:.2f}")\
            .highlight_max(subset=["Avg Stockout Days", "Avg Cost (₹)", "Avg Working Capital (₹)"], props='background-color: #FF4B4B; color: black;')\
            .highlight_min(subset=["Avg Stockout Days", "Avg Cost (₹)", "Avg Working Capital (₹)"], props='background-color: #228B22; color: white;')\
            .highlight_max(subset=["Avg Fill Rate (%)"], props='background-color: #228B22; color: white;')
        st.table(styled_sens)

        # --- SECTION 4: RISK DISTRIBUTIONS ---
        st.divider()
        st.write("### 📈 Risk Distributions")
        r1_c1, r1_c2 = st.columns(2)
        r1_c1.plotly_chart(px.histogram(curr_df, x="fill_rate", title="Fill Rate Distribution", color_discrete_sequence=['#00CC96']), use_container_width=True)
        r1_c2.plotly_chart(px.histogram(curr_df, x="total_cost", title="Cost Distribution", color_discrete_sequence=['#EF553B']), use_container_width=True)

        r2_c1, r2_c2 = st.columns(2)
        r2_c1.plotly_chart(px.histogram(curr_df, x="stockout_days", title="Stockout Severity", color_discrete_sequence=['#FFA15A']), use_container_width=True)
        r2_c2.plotly_chart(px.histogram(curr_df, x="avg_inv", title="Avg Inventory Level", color_discrete_sequence=['#636EFA']), use_container_width=True)

        with st.expander("View Raw Statistical Data"):
            st.dataframe(curr_df.describe().T)

with tab3:
    st.header("🧬 AI Inventory Optimizer")
    
    # --- 1. TARGETS & SEARCH SPACE ---
    c_input1, c_input2 = st.columns(2)
    with c_input1:
        target_fr = st.slider("Target Annual Fill Rate (%)", 85.0, 100.0, 95.0)
        target_so_days = st.number_input("Max Allowed Stockout Days", value=2)
    
    with c_input2:
        num_pop = 20  
        num_gen = 10   
        num_sim = 2000 

    # Statistical Bounds
    avg_ltd = avg_demand * lead_time
    sigma_ltd = std_demand * np.sqrt(lead_time)
    rop_floor, rop_ceil = int(max(0, avg_ltd)), int(avg_ltd + (5.5 * sigma_ltd))

    # --- 2. LIVE TRACKING PLACEHOLDERS ---
    if st.button("🚀 Run Vectorized Optimization"):
        start_time = time.time()
        
        # Placeholders for live updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        table_placeholder = st.empty() # This will show the stepwise progress
        
        # Pre-generate Demand Matrix
        demand_matrix = np.maximum(0, np.random.normal(avg_demand, std_demand, (num_sim, num_days))).round()
        
        # Initialize Population
        bounds = [(rop_floor, rop_ceil), (100, int(avg_demand * 45))]
        pop = [[np.random.randint(b[0], b[1]) for b in bounds] for _ in range(num_pop)]
        
        stepwise_data = [] # To store the best of each generation
        history = []

        for gen in range(num_gen):
            fitness_scores = []
            
            for r_t, q_t in pop:
                # Vectorized Simulation
                inventory = np.full(num_sim, opening_balance, dtype=float)
                arrival_days = np.full(num_sim, -1)
                so_days, total_unmet, h_costs, orders = np.zeros(num_sim), np.zeros(num_sim), np.zeros(num_sim), np.zeros(num_sim)

                for d in range(num_days):
                    arrived = (arrival_days == d)
                    inventory[arrived] += q_t
                    arrival_days[arrived] = -1
                    
                    d_today = demand_matrix[:, d]
                    shortfall = np.maximum(0, d_today - inventory)
                    so_days[shortfall > 0] += 1
                    total_unmet += shortfall
                    inventory = np.maximum(0, inventory - d_today)
                    h_costs += (inventory * (unit_value * holding_cost_rate / 365))
                    
                    reorder_mask = (inventory <= r_t) & (arrival_days == -1)
                    arrival_days[reorder_mask] = d + lead_time
                    orders[reorder_mask] += 1

                fill_rates = (1 - (total_unmet / demand_matrix.sum(axis=1))) * 100
                scenario_costs = h_costs + (orders * ordering_cost)
                
                # Metrics for Fitness
                p99_so, p1_fr, avg_cost = np.percentile(so_days, 99), np.percentile(fill_rates, 1), np.mean(scenario_costs)

                if p99_so <= target_so_days and p1_fr >= target_fr:
                    score = avg_cost
                else:
                    score = avg_cost * 1000 * (1 + (p99_so - target_so_days) + (target_fr - p1_fr))
                
                fitness_scores.append(score)

            # --- RECORD STEPWISE PROGRESS ---
            ranked = [x for _, x in sorted(zip(fitness_scores, pop))]
            best_score = min(fitness_scores)
            is_feasible = "✅ Yes" if best_score < 1e7 else "❌ No"
            
            # Save the best policy of this generation to the log
            stepwise_data.append({
                "Gen": gen + 1,
                "Best ROP": ranked[0][0],
                "Best Qty": ranked[0][1],
                "Annual Cost": f"₹{round(best_score, 0)}" if best_score < 1e7 else "Penalized",
                "Feasible?": is_feasible
            })
            
            # UPDATE LIVE TABLE
            import pandas as pd
            table_placeholder.table(pd.DataFrame(stepwise_data).set_index("Gen"))
            
            history.append(best_score if best_score < 1e7 else None)
            
            # Evolution Step
            new_pop = ranked[:2]
            while len(new_pop) < num_pop:
                p1, p2 = random.sample(ranked[:8], 2)
                child = [int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)]
                if np.random.random() < 0.2:
                    child[0] = np.clip(child[0] + np.random.randint(-30, 30), rop_floor, rop_ceil)
                new_pop.append(child)
            pop = new_pop
            
            progress_bar.progress((gen + 1) / num_gen)
            status_text.text(f"Processing Generation {gen+1} of {num_gen}...")

        # --- FINAL DISPLAY ---
        st.success(f"Optimization finished in {round(time.time()-start_time, 2)}s")
        plot_history = [h for h in history if h is not None]
        if plot_history:
            st.plotly_chart(px.line(y=plot_history, title="Minima Convergence Plot", labels={'y':'Cost ₹','x':'Gen'}, markers=True))
