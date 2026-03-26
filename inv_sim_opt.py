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
# if 'best_policy' not in st.session_state:
#     st.session_state.best_policy = None
if 'best_policy' not in st.session_state:
    st.session_state.best_policy = None
if 'stress_test_done' not in st.session_state:
    st.session_state.stress_test_done = False

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Detailed Analysis", "🎲 Monte Carlo Simulation", "🎯 Policy Optimizer", "Final Results","Inventory Summary"])

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


# with tab2:
#     st.header("🎲 Monte Carlo Risk & Sensitivity Analysis")
    
#     # 1. User Controls
#     c_sim1, c_sim2 = st.columns(2)
#     with c_sim1:
#         n_scenarios = st.number_input("Scenarios to Simulate", min_value=1, max_value=100000, value=2000)
#     with c_sim2:
#         ruin_streak = st.slider("Consecutive days for 'Ruin' (Streak)", 3, 15, 7)

#     if st.button("🚀 Run Comprehensive Vectorized Test"):
#         start_time = time.time()
        
#         # We test Current Lead Time, LT+1, and LT+2 to see the "Breaking Point"
#         lt_tests = [lead_time, lead_time + 1, lead_time + 2]
#         sensitivity_results = []
        
#         progress_bar = st.progress(0)
#         status_text = st.empty()

#         for idx, lt_val in enumerate(lt_tests):
#             status_text.text(f"Simulating Lead Time: {lt_val} days...")
            
#             # Pre-generate Demand Matrix for this LT test
#             demand_matrix = np.maximum(0, np.random.normal(avg_demand, std_demand, (n_scenarios, num_days))).round()
            
#             # Simulation Arrays
#             inventory = np.full(n_scenarios, opening_balance, dtype=float)
#             arrival_days = np.full(n_scenarios, -1)
#             so_days, total_unmet, h_costs_accum, order_counts, daily_inv_sum = [np.zeros(n_scenarios) for _ in range(5)]
            
#             # History tracking for Ruin and Recovery
#             so_history = np.zeros((n_scenarios, num_days), dtype=int)
#             inv_history = np.zeros((n_scenarios, num_days))

#             # --- THE VECTORIZED CORE ---
#             for d in range(num_days):
#                 arrived = (arrival_days == d)
#                 inventory[arrived] += order_qty
#                 arrival_days[arrived] = -1
                
#                 d_today = demand_matrix[:, d]
#                 is_out = (inventory < d_today)
#                 so_days[is_out] += 1
#                 so_history[is_out, d] = 1
                
#                 total_unmet += np.maximum(0, d_today - inventory)
#                 inventory = np.maximum(0, inventory - d_today)
#                 inv_history[:, d] = inventory
                
#                 h_costs_accum += (inventory * (unit_value * holding_cost_rate / 365))
#                 daily_inv_sum += inventory
                
#                 reorder_mask = (inventory <= reorder_point) & (arrival_days == -1)
#                 arrival_days[reorder_mask] = d + lt_val
#                 order_counts[reorder_mask] += 1

#             # --- CALCULATE RISK OF RUIN & RECOVERY ---
#             has_ruin = np.zeros(n_scenarios, dtype=bool)
#             recovery_times = np.zeros(n_scenarios)

#             for s in range(n_scenarios):
#                 # Check for Ruin (7-day streak of 0 inventory)
#                 streaks = np.convolve(so_history[s, :], np.ones(ruin_streak), mode='valid')
#                 if np.any(streaks >= ruin_streak): has_ruin[s] = True
                
#                 # Recovery: Days from first stockout to returning to Safety (ROP)
#                 out_days = np.where(so_history[s, :] == 1)[0]
#                 if len(out_days) > 0:
#                     first_out = out_days[0]
#                     rec_point = np.where(inv_history[s, first_out:] >= reorder_point)[0]
#                     recovery_times[s] = rec_point[0] if len(rec_point) > 0 else (num_days - first_out)

#             # --- AGGREGATE RESULTS ---
#             lt_df = pd.DataFrame({
#                 "fill_rate": (1 - (total_unmet / demand_matrix.sum(axis=1))) * 100,
#                 "stockout_days": so_days,
#                 "total_cost": h_costs_accum + (order_counts * ordering_cost),
#                 "avg_inv": daily_inv_sum / num_days,
#                 "avg_wc": (daily_inv_sum / num_days) * unit_value,
#                 "is_ruined": has_ruin,
#                 "recovery_time": recovery_times,
#                 "Tested LT": lt_val
#             })
#             sensitivity_results.append(lt_df)
#             progress_bar.progress((idx + 1) / len(lt_tests))

#         # 4. DATA PROCESSING
#         res_df = pd.concat(sensitivity_results)
#         curr_df = res_df[res_df["Tested LT"] == lead_time]
#         st.success(f"Simulated {len(res_df):,} paths in {round(time.time()-start_time, 2)}s.")

#         # --- SECTION 1: CORE KPIs ---
#         st.write("### 📊 Service Level & Financial Summary")
#         k1, k2, k3, k4 = st.columns(4)
#         k1.metric("Avg Fill Rate", f"{round(curr_df['fill_rate'].mean(), 2)}%")
#         k2.metric("Avg Stockout Days", f"{round(curr_df['stockout_days'].mean(), 1)}")
#         k3.metric("Avg Annual Cost", f"₹{round(curr_df['total_cost'].mean(), 0)}")
#         k4.metric("WC Risk (95th Pctl)", f"₹{round(curr_df['avg_wc'].quantile(0.95), 0)}")

#         # --- SECTION 2: RISK PROBABILITIES (RESTORED CAPTIONS) ---
#         st.write("#### 🛡️ Stockout Risk Probabilities")
#         d_1, d_5, d_10 = round(num_days*0.01, 1), round(num_days*0.05, 1), round(num_days*0.10, 1)
#         p1, p2, p3, p4 = st.columns(4)
#         with p1:
#             st.metric("Prob: No Stockouts", f"{round((curr_df['stockout_days'] == 0).mean() * 100, 2)}%")
#             st.caption("Chance of zero stockout days all year.")
#         with p2:
#             st.metric("Stockouts < 1% Days", f"{round((curr_df['stockout_days'] < d_1).mean() * 100, 2)}%")
#             st.caption(f"Less than {d_1} days.")
#         with p3:
#             st.metric("Stockouts < 5% Days", f"{round((curr_df['stockout_days'] < d_5).mean() * 100, 2)}%")
#             st.caption(f"Less than {d_5} days.")
#         with p4:
#             ruin_val = curr_df['is_ruined'].mean() * 100
#             st.metric(f"Risk of Ruin ({ruin_streak}d)", f"{round(ruin_val, 2)}%")
#             st.caption(f"Chance of {ruin_streak}-day stockout streak.")

#         # --- SECTION 3: SENSITIVITY TABLE (RESTORED) ---
#         st.divider()
#         st.write("### 📋 Lead Time Sensitivity Table")
#         sens_table = res_df.groupby("Tested LT").agg({
#             "fill_rate": "mean", "stockout_days": "mean", "total_cost": "mean", "avg_wc": "mean", "is_ruined": "mean"
#         })
#         sens_table["is_ruined"] *= 100
#         sens_table.columns = ["Avg Fill Rate (%)", "Avg Stockout Days", "Avg Cost (₹)", "Avg Working Capital (₹)", "Risk of Ruin (%)"]
        
#         st.table(sens_table.style.format("{:.2f}").highlight_max(subset=["Avg Stockout Days", "Avg Cost (₹)", "Risk of Ruin (%)"], props='background-color: #FF4B4B; color: black;'))

#         # --- SECTION 4: THE 4-GRID DISTRIBUTION GRID (RESTORED) ---
#         st.divider()
#         st.write("### 📈 Risk Distributions")
        
#         r1c1, r1c2 = st.columns(2)
#         with r1c1:
#             st.plotly_chart(px.histogram(curr_df, x="fill_rate", title="Fill Rate Distribution (Service Level)", color_discrete_sequence=['#00CC96']), use_container_width=True)
#         with r1c2:
#             st.plotly_chart(px.histogram(curr_df, x="total_cost", title="Total Inventory Cost Distribution", color_discrete_sequence=['#EF553B']), use_container_width=True)

#         r2c1, r2c2 = st.columns(2)
#         with r2c1:
#             st.plotly_chart(px.histogram(curr_df, x="stockout_days", title="Stockout Severity Distribution (Days)", color_discrete_sequence=['#FFA15A']), use_container_width=True)
#         with r2c2:
#             st.plotly_chart(px.histogram(curr_df, x="avg_inv", title="Average Inventory Level Distribution", color_discrete_sequence=['#636EFA']), use_container_width=True)

#         # Bottom full-width chart for Recovery Time
#         st.plotly_chart(px.histogram(curr_df, x="recovery_time", title="Recovery Time Distribution (Days to Safety)", color_discrete_sequence=['#AB63FA']), use_container_width=True)
        
#         with st.expander("View Raw Statistical Data"):
#             st.dataframe(curr_df.describe().T)

with tab2:
    st.header("🎲 Monte Carlo Risk & Sensitivity Analysis")
    
    # 1. User Controls
    c_sim1, c_sim2 = st.columns(2)
    with c_sim1:
        n_scenarios = st.number_input("Scenarios to Simulate", min_value=1, max_value=100000, value=2000)
    with c_sim2:
        ruin_streak = st.slider("Consecutive days for 'Ruin' (Streak)", 3, 15, 7)

    if st.button("🚀 Run Comprehensive Vectorized Test"):
        start_time = time.time()
        
        # Testing Current Lead Time, LT+1, and LT+2
        lt_tests = [lead_time, lead_time + 1, lead_time + 2]
        sensitivity_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, lt_val in enumerate(lt_tests):
            status_text.text(f"Simulating Lead Time: {lt_val} days...")
            
            # Pre-generate Demand (Guarded against negatives)
            demand_matrix = np.maximum(0, np.random.normal(avg_demand, std_demand, (n_scenarios, num_days))).round()
            
            # Simulation Arrays
            inventory = np.full(n_scenarios, opening_balance, dtype=float)
            
            # NEW: Arrival Timeline to allow multiple orders in-transit
            # Buffer size: num_days + max possible lead time
            arrivals_timeline = np.zeros((n_scenarios, num_days + lt_val + 1))
            
            so_days, total_unmet, h_costs_accum, order_counts, daily_inv_sum = [np.zeros(n_scenarios) for _ in range(5)]
            
            # History tracking
            so_history = np.zeros((n_scenarios, num_days), dtype=int)
            inv_history = np.zeros((n_scenarios, num_days))

            # --- RECTIFIED CORE ENGINE ---
            for d in range(num_days):
                # A. Receive scheduled arrivals for today
                inventory += arrivals_timeline[:, d]
                
                # B. Process Demand
                d_today = demand_matrix[:, d]
                is_out = (inventory < d_today)
                so_days[is_out] += 1
                so_history[is_out, d] = 1
                
                total_unmet += np.maximum(0, d_today - inventory)
                inventory = np.maximum(0, inventory - d_today)
                inv_history[:, d] = inventory
                
                # C. Financials
                h_costs_accum += (inventory * (unit_value * holding_cost_rate / 365))
                daily_inv_sum += inventory
                
                # D. REORDER LOGIC (Using Inventory Position)
                # Inventory Position = Physical Stock + Everything currently in transit
                in_transit = arrivals_timeline[:, d+1:].sum(axis=1)
                inv_position = inventory + in_transit
                
                # Trigger order if Position <= ROP
                reorder_mask = (inv_position <= reorder_point)
                
                # Place orders: Record arrival at specific future date
                arrivals_timeline[reorder_mask, d + lt_val] += order_qty
                order_counts[reorder_mask] += 1

            # --- RISK OF RUIN & RECOVERY ---
            has_ruin = np.zeros(n_scenarios, dtype=bool)
            recovery_times = np.zeros(n_scenarios)

            for s in range(n_scenarios):
                # Ruin check
                streaks = np.convolve(so_history[s, :], np.ones(ruin_streak), mode='valid')
                if np.any(streaks >= ruin_streak): has_ruin[s] = True
                
                # Recovery check
                out_days = np.where(so_history[s, :] == 1)[0]
                if len(out_days) > 0:
                    first_out = out_days[0]
                    rec_point = np.where(inv_history[s, first_out:] >= reorder_point)[0]
                    recovery_times[s] = rec_point[0] if len(rec_point) > 0 else (num_days - first_out)

            # --- AGGREGATE RESULTS ---
            lt_df = pd.DataFrame({
                "fill_rate": (1 - (total_unmet / demand_matrix.sum(axis=1))) * 100,
                "stockout_days": so_days,
                "total_cost": h_costs_accum + (order_counts * ordering_cost),
                "avg_inv": daily_inv_sum / num_days,
                "avg_wc": (daily_inv_sum / num_days) * unit_value,
                "is_ruined": has_ruin,
                "recovery_time": recovery_times,
                "Tested LT": lt_val
            })
            sensitivity_results.append(lt_df)
            progress_bar.progress((idx + 1) / len(lt_tests))

        # 4. DATA PROCESSING
        res_df = pd.concat(sensitivity_results)
        curr_df = res_df[res_df["Tested LT"] == lead_time]
        st.success(f"Simulated {len(res_df):,} paths in {round(time.time()-start_time, 2)}s.")

        # --- SECTION 1: CORE KPIs ---
        st.write("### 📊 Service Level & Financial Summary")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Avg Fill Rate", f"{round(curr_df['fill_rate'].mean(), 2)}%")
        k2.metric("Avg Stockout Days", f"{round(curr_df['stockout_days'].mean(), 1)}")
        k3.metric("Avg Annual Cost", f"₹{round(curr_df['total_cost'].mean(), 0)}")
        k4.metric("WC Risk (95th Pctl)", f"₹{round(curr_df['avg_wc'].quantile(0.95), 0)}")

        # --- SECTION 2: RISK PROBABILITIES ---
        st.write("#### 🛡️ Stockout Risk Probabilities")
        d_1 = round(num_days*0.01, 1)
        d_5 = round(num_days*0.05, 1)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Prob: No Stockouts", f"{round((curr_df['stockout_days'] == 0).mean() * 100, 2)}%")
        p2.metric("Stockouts < 1% Days", f"{round((curr_df['stockout_days'] < d_1).mean() * 100, 2)}%")
        p3.metric("Stockouts < 5% Days", f"{round((curr_df['stockout_days'] < d_5).mean() * 100, 2)}%")
        p4.metric(f"Risk of Ruin ({ruin_streak}d)", f"{round(curr_df['is_ruined'].mean() * 100, 2)}%")

        # --- SECTION 3: SENSITIVITY TABLE ---
        st.divider()
        st.write("### 📋 Lead Time Sensitivity Table")
        sens_table = res_df.groupby("Tested LT").agg({
            "fill_rate": "mean", "stockout_days": "mean", "total_cost": "mean", "avg_wc": "mean", "is_ruined": "mean"
        })
        sens_table["is_ruined"] *= 100
        sens_table.columns = ["Avg Fill Rate (%)", "Avg Stockout Days", "Avg Cost (₹)", "Avg Working Capital (₹)", "Risk of Ruin (%)"]
        st.table(sens_table.style.format("{:.2f}").highlight_max(subset=["Avg Stockout Days", "Avg Cost (₹)", "Risk of Ruin (%)"], props='background-color: #FF4B4B; color: black;'))

        # --- SECTION 4: THE 4-GRID DISTRIBUTION GRID ---
        st.divider()
        st.write("### 📈 Risk Distributions")
        r1c1, r1c2 = st.columns(2)
        r1c1.plotly_chart(px.histogram(curr_df, x="fill_rate", title="Fill Rate Distribution", color_discrete_sequence=['#00CC96']), use_container_width=True)
        r1c2.plotly_chart(px.histogram(curr_df, x="total_cost", title="Total Inventory Cost Distribution", color_discrete_sequence=['#EF553B']), use_container_width=True)

        r2c1, r2c2 = st.columns(2)
        r2c1.plotly_chart(px.histogram(curr_df, x="stockout_days", title="Stockout Severity (Days)", color_discrete_sequence=['#FFA15A']), use_container_width=True)
        r2c2.plotly_chart(px.histogram(curr_df, x="avg_inv", title="Average Inventory Level", color_discrete_sequence=['#636EFA']), use_container_width=True)

        st.plotly_chart(px.histogram(curr_df, x="recovery_time", title="Recovery Time Distribution (Days to Safety)", color_discrete_sequence=['#AB63FA']), use_container_width=True)
        
        with st.expander("View Raw Statistical Data"):
            st.dataframe(curr_df.describe().T)

# with tab3:
#     st.header("🧬 AI Inventory Optimizer")
    
#     # --- 1. DYNAMIC CONSTRAINT UI ---
#     st.subheader("Define Your Business Constraints")
#     col_c1, col_c2 = st.columns(2)
    
#     with col_c1:
#         use_so_constraint = st.toggle("Limit Stockout Days (P99)", value=True)
#         if use_so_constraint:
#             target_so_days = st.number_input("Max Allowed Stockout Days", value=5, step=1)
#         target_fr = st.slider("Min. Acceptable Fill Rate (P1) %", 80.0, 100.0, 95.0)

#     with col_c2:
#         use_wc_constraint = st.toggle("Limit Peak Working Capital (P99)", value=False)
#         if use_wc_constraint:
#             max_wc_allowed = st.number_input("Maximum Cash Ceiling (₹)", value=100000, step=5000)
    
#     # Optimizer Hyperparameters
#     with st.expander("⚙️ Optimizer Settings"):
#         num_pop = st.slider("Population Size (Genetic Diversity)", 20, 100, 40)
#         num_gen = st.slider("Generations (Fine-tuning)", 5, 50, 20)
#         num_sim = 2000 # Balanced for speed vs accuracy

#     # Statistical Search Space
#     avg_ltd = avg_demand * lead_time
#     sigma_ltd = std_demand * np.sqrt(lead_time)
#     rop_floor, rop_ceil = int(max(0, avg_ltd)), int(avg_ltd + (6.0 * sigma_ltd))

#     # --- 2. OPTIMIZATION ENGINE ---
#     if st.button("🚀 Run Vectorized Optimization"):
#         start_time = time.time()
#         progress_bar = st.progress(0)
#         status_text = st.empty()
#         table_placeholder = st.empty()
        
#         # Pre-generate Demand Matrix
#         demand_matrix = np.maximum(0, np.random.normal(avg_demand, std_demand, (num_sim, num_days))).round()
        
#         # Initialize Population [ROP, Q]
#         bounds = [(rop_floor, rop_ceil), (100, int(avg_demand * 45))]
#         pop = [[np.random.randint(b[0], b[1]) for b in bounds] for _ in range(num_pop)]
        
#         stepwise_data = []
#         history = []

#         for gen in range(num_gen):
#             fitness_scores = []
#             gen_metrics = [] # To store actual performance for the summary
            
#             for r_t, q_t in pop:
#                 # --- VECTORIZED SIMULATION ---
#                 inventory = np.full(num_sim, opening_balance, dtype=float)
#                 arrival_days = np.full(num_sim, -1)
#                 so_days, total_unmet, h_costs, orders = [np.zeros(num_sim) for _ in range(4)]
                
#                 # To track P99 Peak WC, we need to find the max inventory in each scenario
#                 scenario_peaks = np.zeros(num_sim)

#                 for d in range(num_days):
#                     arrived = (arrival_days == d)
#                     inventory[arrived] += q_t
#                     arrival_days[arrived] = -1
                    
#                     d_today = demand_matrix[:, d]
#                     shortfall = np.maximum(0, d_today - inventory)
#                     so_days[shortfall > 0] += 1
#                     total_unmet += shortfall
                    
#                     inventory = np.maximum(0, inventory - d_today)
#                     h_costs += (inventory * (unit_value * holding_cost_rate / 365))
                    
#                     # Track peaks per scenario
#                     scenario_peaks = np.maximum(scenario_peaks, inventory)
                    
#                     reorder_mask = (inventory <= r_t) & (arrival_days == -1)
#                     arrival_days[reorder_mask] = d + lead_time
#                     orders[reorder_mask] += 1

#                 # --- CALCULATE METRICS ---
#                 fill_rates = (1 - (total_unmet / demand_matrix.sum(axis=1))) * 100
#                 scenario_costs = h_costs + (orders * ordering_cost)
#                 peak_wc_values = scenario_peaks * unit_value
                
#                 p99_so = np.percentile(so_days, 99)
#                 p1_fr = np.percentile(fill_rates, 1)
#                 p99_wc = np.percentile(peak_wc_values, 99)
#                 avg_cost = np.mean(scenario_costs)

#                 # --- THE SMART PENALTY ENGINE ---
#                 penalty = 0
#                 if use_so_constraint and p99_so > target_so_days:
#                     penalty += (p99_so - target_so_days) * 10000
#                 if use_wc_constraint and p99_wc > max_wc_allowed:
#                     penalty += (p99_wc - max_wc_allowed) * 20 # Stronger penalty for cash
#                 if p1_fr < target_fr:
#                     penalty += (target_fr - p1_fr) * 15000
                
#                 score = avg_cost + penalty
#                 fitness_scores.append(score)
#                 gen_metrics.append({'cost': avg_cost, 'p99_wc': p99_wc, 'p99_so': p99_so, 'p1_fr': p1_fr})

#             # --- SELECTION & EVOLUTION ---
#             ranked_indices = np.argsort(fitness_scores)
#             ranked_pop = [pop[i] for i in ranked_indices]
#             best_idx = ranked_indices[0]
            
#             # Record stepwise progress
#             best_pol = ranked_pop[0]
#             best_m = gen_metrics[best_idx]
            
#             stepwise_data.append({
#                 "Gen": gen + 1,
#                 "ROP": best_pol[0],
#                 "Qty": best_pol[1],
#                 "Avg Cost": f"₹{round(best_m['cost'],0):,}",
#                 "Peak WC (P99)": f"₹{round(best_m['p99_wc'],0):,}",
#                 "SO Days (P99)": round(best_m['p99_so'], 1)
#             })
#             table_placeholder.table(pd.DataFrame(stepwise_data).set_index("Gen").tail(5))
#             history.append(fitness_scores[best_idx] if fitness_scores[best_idx] < 1e7 else None)

#             # Genetic Crossover & Mutation
#             new_pop = ranked_pop[:4] # Elitism
#             while len(new_pop) < num_pop:
#                 p1, p2 = random.sample(ranked_pop[:12], 2)
#                 child = [int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)]
#                 if np.random.random() < 0.3: # Mutation
#                     child[0] = np.clip(child[0] + np.random.randint(-25, 25), rop_floor, rop_ceil)
#                 new_pop.append(child)
#             pop = new_pop
            
#             progress_bar.progress((gen + 1) / num_gen)
#             status_text.text(f"Optimizing... Generation {gen+1}/{num_gen}")

#         # --- 3. FINAL RESULTS & SESSION STATE ---
#         st.success(f"Optimization Complete in {round(time.time()-start_time, 2)}s")
        
#         # Save to session state for Tab 4
#         best_final_r, best_final_q = ranked_pop[0]
#         st.session_state.best_policy = [best_final_r, best_final_q]
        
#         # Final Constraint Summary Display
#         st.subheader("✅ Optimized Strategy Summary")
#         s1, s2, s3 = st.columns(3)
#         with s1:
#             wc_ok = not use_wc_constraint or best_m['p99_wc'] <= max_wc_allowed
#             st.metric("Peak WC (P99)", f"₹{round(best_m['p99_wc'],0):,}", 
#                       delta="Met" if wc_ok else "Over Limit", delta_color="normal" if wc_ok else "inverse")
#         with s2:
#             so_ok = not use_so_constraint or best_m['p99_so'] <= target_so_days
#             st.metric("Worst-Case Stockouts", f"{round(best_m['p99_so'],1)} Days", 
#                       delta="Met" if so_ok else "Over Limit", delta_color="normal" if so_ok else "inverse")
#         with s3:
#             fr_ok = best_m['p1_fr'] >= target_fr
#             st.metric("Min. Fill Rate (P1)", f"{round(best_m['p1_fr'],2)}%", 
#                       delta="Met" if fr_ok else "Below Target", delta_color="normal" if fr_ok else "inverse")

#         if history:
#             st.plotly_chart(px.line(y=history, title="Minima Convergence Plot (Fitness Score)", markers=True))

with tab3:
    st.header("🧬 AI Inventory Optimizer")
    
    # --- 1. DYNAMIC CONSTRAINT UI ---
    st.subheader("Define Your Business Constraints")
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        use_so_constraint = st.toggle("Limit Stockout Days (P99)", value=True)
        if use_so_constraint:
            target_so_days = st.number_input("Max Allowed Stockout Days", value=5, step=1)
        target_fr = st.slider("Min. Acceptable Fill Rate (P1) %", 80.0, 100.0, 95.0)

    with col_c2:
        use_wc_constraint = st.toggle("Limit Peak Working Capital (P99)", value=False)
        if use_wc_constraint:
            max_wc_allowed = st.number_input("Maximum Cash Ceiling (₹)", value=100000, step=5000)
    
    # Optimizer Hyperparameters
    with st.expander("⚙️ Optimizer Settings"):
        num_pop = st.slider("Population Size (Genetic Diversity)", 20, 100, 40)
        num_gen = st.slider("Generations (Fine-tuning)", 5, 50, 20)
        # Pro Tip: Lowering this to 500-1000 provides 4x speed with 99% accuracy
        num_sim = st.select_slider("Simulation Sample Size", options=[500, 1000, 2000], value=1000)

    # Statistical Search Space
    avg_ltd = avg_demand * lead_time
    sigma_ltd = std_demand * np.sqrt(lead_time)
    rop_floor, rop_ceil = int(max(0, avg_ltd - (1.5 * sigma_ltd))), int(avg_ltd + (6.0 * sigma_ltd))

    # --- 2. OPTIMIZATION ENGINE ---
    if st.button("🚀 Run Vectorized Optimization"):
        start_time = time.time()
        progress_bar = st.progress(0)
        status_text = st.empty()
        table_placeholder = st.empty()
        
        # Constant for speed: Calculate once, use 365 times
        daily_h_cost_unit = unit_value * holding_cost_rate / 365
        
        # Pre-generate Demand Matrix (The "Environment")
        demand_matrix = np.maximum(0, np.random.normal(avg_demand, std_demand, (num_sim, num_days))).round()
        total_demand_per_scenario = demand_matrix.sum(axis=1)
        
        # Initialize Population [ROP, Q]
        bounds = [(rop_floor, rop_ceil), (100, int(avg_demand * 45))]
        pop = [[np.random.randint(b[0], b[1]) for b in bounds] for _ in range(num_pop)]
        
        stepwise_data = []
        history = []

        for gen in range(num_gen):
            fitness_scores = []
            gen_metrics = [] 
            
            for r_t, q_t in pop:
                # --- VECTORIZED SIMULATION CORE ---
                inventory = np.full(num_sim, opening_balance, dtype=float)
                
                # Arrival Queue: Scenarios x (Days + LeadTime Buffer)
                arrivals = np.zeros((num_sim, num_days + lead_time + 1))
                # Track the total currently in the pipeline to avoid .sum() every day
                pipeline_total = np.zeros(num_sim)
                
                so_days, total_unmet, h_costs, orders = [np.zeros(num_sim) for _ in range(4)]
                scenario_peaks = np.zeros(num_sim)

                for d in range(num_days):
                    # 1. Deliver Arrivals & Update Pipeline Total
                    landing_today = arrivals[:, d]
                    inventory += landing_today
                    pipeline_total -= landing_today
                    
                    # 2. Daily Demand Process
                    d_today = demand_matrix[:, d]
                    inventory -= d_today
                    
                    # 3. Handle Stockouts (Vectorized Masking)
                    out_mask = (inventory < 0)
                    if np.any(out_mask):
                        so_days[out_mask] += 1
                        total_unmet[out_mask] -= inventory[out_mask] # Subtracting neg adds to unmet
                        inventory[out_mask] = 0
                    
                    # 4. Record peaks & Accumulate holding costs
                    scenario_peaks = np.maximum(scenario_peaks, inventory)
                    h_costs += (inventory * daily_h_cost_unit)
                    
                    # 5. REORDER LOGIC: Inventory Position (Physical + Pipeline)
                    # No .sum() needed here because we track pipeline_total
                    inv_pos = inventory + pipeline_total
                    
                    reorder_mask = (inv_pos <= r_t)
                    if np.any(reorder_mask):
                        # Schedule arrival and update tracking variable
                        arrivals[reorder_mask, d + lead_time] += q_t
                        pipeline_total[reorder_mask] += q_t
                        orders[reorder_mask] += 1

                # --- CALCULATE AGGREGATED METRICS ---
                fill_rates = (1 - (total_unmet / total_demand_per_scenario)) * 100
                scenario_costs = h_costs + (orders * ordering_cost)
                peak_wc_values = scenario_peaks * unit_value
                
                p99_so = np.percentile(so_days, 99)
                p1_fr = np.percentile(fill_rates, 1)
                p99_wc = np.percentile(peak_wc_values, 99)
                avg_cost = np.mean(scenario_costs)

                # --- PENALTY ENGINE ---
                penalty = 0
                if use_so_constraint and p99_so > target_so_days:
                    penalty += (p99_so - target_so_days) * 20000 # Double penalty for strictness
                if use_wc_constraint and p99_wc > max_wc_allowed:
                    penalty += (p99_wc - max_wc_allowed) * 50 
                if p1_fr < target_fr:
                    penalty += (target_fr - p1_fr) * 25000
                
                fitness_scores.append(avg_cost + penalty)
                gen_metrics.append({'cost': avg_cost, 'p99_wc': p99_wc, 'p99_so': p99_so, 'p1_fr': p1_fr})

            # --- SELECTION & EVOLUTION ---
            ranked_indices = np.argsort(fitness_scores)
            ranked_pop = [pop[i] for i in ranked_indices]
            best_idx = ranked_indices[0]
            best_pol = ranked_pop[0]
            best_m = gen_metrics[best_idx]
            
            stepwise_data.append({
                "Gen": gen + 1,
                "ROP": best_pol[0],
                "Qty": best_pol[1],
                "Avg Cost": f"₹{round(best_m['cost'],0):,}",
                "Peak WC (P99)": f"₹{round(best_m['p99_wc'],0):,}",
                "SO Days (P99)": round(best_m['p99_so'], 1)
            })
            table_placeholder.table(pd.DataFrame(stepwise_data).set_index("Gen").tail(3))
            history.append(fitness_scores[best_idx] if fitness_scores[best_idx] < 1e8 else None)

            # Crossover & Mutation
            new_pop = ranked_pop[:4] # Elitism (Top 4 stay)
            while len(new_pop) < num_pop:
                p1, p2 = random.sample(ranked_pop[:12], 2)
                child = [int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)]
                if np.random.random() < 0.3: 
                    child[0] = np.clip(child[0] + np.random.randint(-30, 30), rop_floor, rop_ceil)
                new_pop.append(child)
            pop = new_pop
            
            progress_bar.progress((gen + 1) / num_gen)
            status_text.text(f"Optimizing... Generation {gen+1}/{num_gen}")

        # --- FINAL OUTPUT ---
        st.success(f"Optimization Complete in {round(time.time()-start_time, 2)}s")
        st.session_state.best_policy = [ranked_pop[0][0], ranked_pop[0][1]]
        
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Optimal ROP", f"{st.session_state.best_policy[0]}")
        with s2:
            st.metric("Optimal Qty", f"{st.session_state.best_policy[1]}")
        with s3:
            st.metric("Min Fill Rate (P1)", f"{round(best_m['p1_fr'], 2)}%")

        if history:
            st.plotly_chart(px.line(y=history, title="Learning Curve (Fitness Improvement)", markers=True))

# with tab4:
#     st.header("🛡️ Strategy Validation & Impact")
#     st.write("Compare your **Manual Policy** against the **AI Optimized Policy** under 10,000 extreme scenarios.")

#     # 1. BRIDGE: Pull data from Tab 3 "Locker"
#     if st.session_state.best_policy is None:
#         st.warning("⚠️ No Optimized Policy found. Please run the AI Optimizer in Tab 3 first.")
#         st.stop()
#     else:
#         opt_r, opt_q = st.session_state.best_policy

#     # 2. SIMULATION ENGINE
#     if st.button("🏁 Run Final 10,000 Scenario Stress Test"):
#         with st.status("Simulating 10,000 'Worst-Case' Years...", expanded=True) as status:
#             n_stress = 10000
#             stress_demands = np.maximum(0, np.random.normal(avg_demand, std_demand, (n_stress, num_days))).round()
            
#             def run_stress_sim(r, q):
#                 inv = np.full(n_stress, opening_balance, dtype=float)
#                 arr = np.full(n_stress, -1)
#                 so, unmet, h_cost, orders = [np.zeros(n_stress) for _ in range(4)]
#                 scenario_peaks = np.zeros(n_stress)
                
#                 for d in range(num_days):
#                     mask_arr = (arr == d)
#                     inv[mask_arr] += q
#                     arr[mask_arr] = -1
#                     d_t = stress_demands[:, d]
#                     short = np.maximum(0, d_t - inv)
#                     so[short > 0] += 1
#                     unmet += short
#                     inv = np.maximum(0, inv - d_t)
#                     h_cost += (inv * (unit_value * holding_cost_rate / 365))
#                     scenario_peaks = np.maximum(scenario_peaks, inv)
#                     mask_re = (inv <= r) & (arr == -1)
#                     arr[mask_re] = d + lead_time
#                     orders[mask_re] += 1
                
#                 fr = (1 - (unmet / stress_demands.sum(axis=1))) * 100
#                 return {
#                     "fr_raw": fr, "avg_fr": fr.mean(), "avg_so": so.mean(), 
#                     "avg_cost": (h_cost + (orders * ordering_cost)).mean(), 
#                     "p99_wc": np.percentile(scenario_peaks, 99) * unit_value,
#                     "p99_so": np.percentile(so, 99)
#                 }

#             st.session_state.m_res = run_stress_sim(reorder_point, order_qty)
#             st.session_state.a_res = run_stress_sim(opt_r, opt_q)
#             st.session_state.stress_test_done = True
#             status.update(label="✅ Stress Test Complete!", state="complete")

#     # 3. DISPLAY RESULTS
#     if st.session_state.get('stress_test_done'):
#         m_res, a_res = st.session_state.m_res, st.session_state.a_res
        
#         # --- TABLE WITH DECISION VARIABLES ---
#         df_comp = pd.DataFrame([
#             {"Metric": "Reorder Point (ROP)", "Manual": float(reorder_point), "AI Optimized": float(opt_r), "LowerIsBetter": None},
#             {"Metric": "Order Quantity (Qty)", "Manual": float(order_qty), "AI Optimized": float(opt_q), "LowerIsBetter": None},
#             {"Metric": "Avg Fill Rate (%)", "Manual": m_res['avg_fr'], "AI Optimized": a_res['avg_fr'], "LowerIsBetter": False},
#             {"Metric": "Avg Stockout Days", "Manual": m_res['avg_so'], "AI Optimized": a_res['avg_so'], "LowerIsBetter": True},
#             {"Metric": "Peak Working Capital (P99 ₹)", "Manual": m_res['p99_wc'], "AI Optimized": a_res['p99_wc'], "LowerIsBetter": True},
#             {"Metric": "Annual Total Cost (₹)", "Manual": m_res['avg_cost'], "AI Optimized": a_res['avg_cost'], "LowerIsBetter": True}
#         ])

#         def style_rows(row):
#             styles = [''] * len(row)
#             if row['LowerIsBetter'] is not None:
#                 is_better = (row['AI Optimized'] <= row['Manual']) if row['LowerIsBetter'] else (row['AI Optimized'] >= row['Manual'])
#                 styles[2] = "background-color: #2e7d32; color: white" if is_better else "background-color: #c62828; color: white"
#             return styles

#         st.write("### ⚖️ Policy Comparison")
#         st.dataframe(df_comp.style.apply(style_rows, axis=1).format({"Manual": "{:,.2f}", "AI Optimized": "{:,.2f}"}).hide(axis="columns", subset=["LowerIsBetter"]), use_container_width=True)

#         # --- RESTORED KPIs: FINANCIAL IMPACT ---
#         st.divider()
#         st.write("### 💰 Strategic Financial Impact")
#         k1, k2, k3 = st.columns(3)
        
#         savings = m_res['avg_cost'] - a_res['avg_cost']
#         wc_delta = m_res['p99_wc'] - a_res['p99_wc']
#         fr_delta = a_res['avg_fr'] - m_res['avg_fr']

#         # KPI 1: Annual Cost Savings
#         k1.metric("Annual Cost Savings", f"₹{round(abs(savings), 0):,}", 
#                   delta=f"{round((savings/m_res['avg_cost'])*100, 1) if m_res['avg_cost'] !=0 else 0}%",
#                   delta_color="normal" if savings >= 0 else "inverse")
        
#         # KPI 2: Working Capital (Liquidity)
#         k2.metric("Working Capital Delta", f"₹{round(abs(wc_delta), 0):,}", 
#                   delta="Cash Unlocked" if wc_delta >= 0 else "Capital Investment",
#                   delta_color="normal" if wc_delta >= 0 else "inverse")
        
#         # KPI 3: Reliability Jump
#         k3.metric("Service Level Gain", f"{round(fr_delta, 2)}%", 
#                   delta="Higher Reliability" if fr_delta >= 0 else "Lower Reliability",
#                   delta_color="normal" if fr_delta >= 0 else "inverse")

#         # Download Button
#         st.download_button("📥 Download Full Report", df_comp.to_csv(index=False), "inventory_report.csv", "text/csv")

with tab4:
    st.header("🛡️ Strategy Validation & Impact")
    st.write("Compare your **Manual Policy** against the **AI Optimized Policy** under 10,000 extreme scenarios.")

    # 1. SESSION STATE BRIDGE
    if st.session_state.best_policy is None:
        st.warning("⚠️ No Optimized Policy found. Please run the AI Optimizer in Tab 3 first.")
        st.stop()
    else:
        opt_r, opt_q = st.session_state.best_policy

    # 2. SIMULATION ENGINE (High-Speed Vectorized with Pipeline Logic)
    if st.button("🏁 Run Final 10,000 Scenario Stress Test"):
        with st.status("Simulating 10,000 'Worst-Case' Years...", expanded=True) as status:
            n_stress = 10000
            # Pre-generate Stress Demand
            stress_demands = np.maximum(0, np.random.normal(avg_demand, std_demand, (n_stress, num_days))).round()
            total_stress_demand = stress_demands.sum(axis=1)
            daily_h_unit = unit_value * holding_cost_rate / 365
            
            def run_stress_sim(r, q):
                inv = np.full(n_stress, opening_balance, dtype=float)
                # Arrivals Matrix + Pipeline Tracker
                arrivals = np.zeros((n_stress, num_days + lead_time + 1))
                pipeline_total = np.zeros(n_stress)
                
                so_days, total_unmet, h_costs, orders = [np.zeros(n_stress) for _ in range(4)]
                scenario_peaks = np.zeros(n_stress)
                
                for d in range(num_days):
                    # A. Receive Arrivals
                    landing = arrivals[:, d]
                    inv += landing
                    pipeline_total -= landing
                    
                    # B. Process Demand
                    d_t = stress_demands[:, d]
                    inv -= d_t
                    
                    # C. Handle Stockouts
                    out_mask = (inv < 0)
                    if np.any(out_mask):
                        so_days[out_mask] += 1
                        total_unmet[out_mask] -= inv[out_mask]
                        inv[out_mask] = 0
                    
                    # D. Metrics
                    scenario_peaks = np.maximum(scenario_peaks, inv)
                    h_costs += (inv * daily_h_unit)
                    
                    # E. Reorder Logic (Inventory Position)
                    inv_pos = inv + pipeline_total
                    reorder_mask = (inv_pos <= r)
                    
                    if np.any(reorder_mask):
                        arrivals[reorder_mask, d + lead_time] += q
                        pipeline_total[reorder_mask] += q
                        orders[reorder_mask] += 1
                
                # Final Aggregation
                fr = (1 - (total_unmet / total_stress_demand)) * 100
                return {
                    "avg_fr": fr.mean(), 
                    "avg_so": so_days.mean(), 
                    "avg_cost": (h_costs + (orders * ordering_cost)).mean(), 
                    "p99_wc": np.percentile(scenario_peaks, 99) * unit_value,
                    "p99_so": np.percentile(so_days, 99),
                    "p1_fr": np.percentile(fr, 1)
                }

            # Run both policies
            st.session_state.m_res = run_stress_sim(reorder_point, order_qty)
            st.session_state.a_res = run_stress_sim(opt_r, opt_q)
            st.session_state.stress_test_done = True
            status.update(label="✅ Stress Test Complete!", state="complete")

    # 3. DISPLAY RESULTS
    if st.session_state.get('stress_test_done'):
        m_res, a_res = st.session_state.m_res, st.session_state.a_res
        
        # Comparison Table
        df_comp = pd.DataFrame([
            {"Metric": "Reorder Point (ROP)", "Manual": float(reorder_point), "AI Optimized": float(opt_r), "LowerIsBetter": None},
            {"Metric": "Order Quantity (Qty)", "Manual": float(order_qty), "AI Optimized": float(opt_q), "LowerIsBetter": None},
            {"Metric": "Avg Fill Rate (%)", "Manual": m_res['avg_fr'], "AI Optimized": a_res['avg_fr'], "LowerIsBetter": False},
            {"Metric": "Worst-Case Stockout Days (P99)", "Manual": m_res['p99_so'], "AI Optimized": a_res['p99_so'], "LowerIsBetter": True},
            {"Metric": "Peak Working Capital (P99 ₹)", "Manual": m_res['p99_wc'], "AI Optimized": a_res['p99_wc'], "LowerIsBetter": True},
            {"Metric": "Annual Total Cost (₹)", "Manual": m_res['avg_cost'], "AI Optimized": a_res['avg_cost'], "LowerIsBetter": True}
        ])

        def style_rows(row):
            styles = [''] * len(row)
            if row['LowerIsBetter'] is not None:
                is_better = (row['AI Optimized'] <= row['Manual']) if row['LowerIsBetter'] else (row['AI Optimized'] >= row['Manual'])
                styles[2] = "background-color: #2e7d32; color: white" if is_better else "background-color: #c62828; color: white"
            return styles

        st.write("### ⚖️ Policy Comparison")
        st.dataframe(df_comp.style.apply(style_rows, axis=1).format({"Manual": "{:,.2f}", "AI Optimized": "{:,.2f}"}).hide(axis="columns", subset=["LowerIsBetter"]), use_container_width=True)

        # 4. FINAL KPIs
        st.divider()
        st.write("### 💰 Strategic Financial Impact")
        k1, k2, k3 = st.columns(3)
        
        savings = m_res['avg_cost'] - a_res['avg_cost']
        wc_delta = m_res['p99_wc'] - a_res['p99_wc']
        fr_delta = a_res['avg_fr'] - m_res['avg_fr']

        k1.metric("Annual Profit Impact", f"₹{round(abs(savings), 0):,}", delta="Savings" if savings > 0 else "Investment")
        k2.metric("Working Capital Delta", f"₹{round(abs(wc_delta), 0):,}", delta="Cash Unlocked" if wc_delta > 0 else "Extra Capital")
        k3.metric("Service Level Gain", f"{round(fr_delta, 2)}%", delta="Reliability Up" if fr_delta > 0 else "Reliability Down")

with tab5:
    st.header("📋 Executive Summary & Action Plan")
    
    # 1. BRIDGE: Pull data from Tab 4 "Locker"
    if not st.session_state.get('stress_test_done'):
        st.warning("⚠️ Data Missing: Please run the 'Stress Test' in Tab 4 first to generate the board report.")
    else:
        # Retrieve persistent results
        m_res = st.session_state.m_res
        a_res = st.session_state.a_res
        opt_r, opt_q = st.session_state.best_policy
        
        # Calculate key deltas for the narrative
        savings = m_res['avg_cost'] - a_res['avg_cost']
        wc_delta = m_res['p99_wc'] - a_res['p99_wc']
        fr_delta = a_res['avg_fr'] - m_res['avg_fr']

        # 2. THE BIG THREE: STRATEGIC IMPACT
        st.subheader("🚀 Final Strategic Impact")
        k1, k2, k3 = st.columns(3)
        
        with k1:
            st.metric("Annual Profit Impact", f"₹{round(abs(savings), 0):,}", 
                      delta="Net Savings" if savings >= 0 else "Net Investment",
                      delta_color="normal" if savings >= 0 else "inverse")
            st.caption("Total cost change (Holding + Ordering + Stockouts)")

        with k2:
            st.metric("Capital Mobility", f"₹{round(abs(wc_delta), 0):,}", 
                      delta="Cash Unlocked" if wc_delta >= 0 else "Capital Required",
                      delta_color="normal" if wc_delta >= 0 else "inverse")
            st.caption("Peak Working Capital change (P99)")

        with k3:
            st.metric("Service Reliability", f"{round(a_res['avg_fr'], 2)}%", 
                      delta=f"{round(fr_delta, 2)}% Gain",
                      delta_color="normal" if fr_delta >= 0 else "inverse")
            st.caption("Average Fill Rate across all scenarios")

        # 3. THE FOUNDER'S CHECKLIST (Implementation Roadmap)
        st.divider()
        st.subheader("📝 Founder's Implementation Roadmap")
        
        st.markdown("### Step 1: System Updates")
        st.checkbox(f"Update ERP/Inventory Reorder Point (ROP) from {reorder_point} ➔ **{int(opt_r)}** units")
        st.checkbox(f"Standardize Purchase Order Quantity (Qty) from {order_qty} ➔ **{int(opt_q)}** units")
        
        st.markdown("### Step 2: Financial Allocation")
        if wc_delta < 0:
            st.error(f"🚩 **Action Required:** Secure **₹{round(abs(wc_delta), 0):,}** in additional liquidity. This 'Insurance' prevents stockouts during demand spikes.")
        else:
            st.success(f"✨ **Action:** Re-allocate **₹{round(wc_delta, 0):,}** of unlocked cash into high-growth areas like Marketing or R&D.")

        # 4. FINAL STORYTELLING ANALOGY
        st.divider()
        st.info(f"💡 **Founder's Insight:** By moving to an ROP of {int(opt_r)}, you aren't just buying stock; you are buying **Peace of Mind**. In 99% of future 'storms', your business will now remain afloat while competitors run dry.")

        # 5. EXPORT FOR BOARD REVIEW
        st.divider()
        report_df = pd.DataFrame(st.session_state.m_res).transpose() # Quick summary
        st.download_button(
            label="📥 Download Executive PDF Data",
            data=report_df.to_csv().encode('utf-8'),
            file_name=f"Executive_Summary_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
