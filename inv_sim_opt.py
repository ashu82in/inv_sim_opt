import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(layout="wide")
st.title("Inventory Policy Simulator")

# =========================================================
# TABS
# =========================================================
tab1, tab2 = st.tabs(["Simulation", "Scenario Analysis"])

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
if st.sidebar.button("Reset Demand Scenario"):
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
# FIFO SIMULATION
# =========================================================
def run_simulation(demand, reorder_point, order_qty):

    inventory_layers = [{"qty": opening_balance, "age": 0}]
    pipeline_orders = []
    data = []
    aging_data = []

    for day in range(num_days):

        for layer in inventory_layers:
            layer["age"] += 1

        for order in pipeline_orders.copy():
            if order[0] == day:
                inventory_layers.append({"qty": order[1], "age": 0})
                pipeline_orders.remove(order)

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
# METRICS FUNCTION
# =========================================================
def run_simulation_metrics(demand, reorder_point, order_qty):

    inventory_layers = [{"qty": opening_balance, "age": 0}]
    pipeline_orders = []
    wc_list = []

    for day in range(num_days):

        for l in inventory_layers:
            l["age"] += 1

        for order in pipeline_orders.copy():
            if order[0] == day:
                inventory_layers.append({"qty": order[1], "age": 0})
                pipeline_orders.remove(order)

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

        if inventory_position < reorder_point:
            pipeline_orders.append((day + lead_time, order_qty))

        wc_list.append(inventory_position * unit_value)

    return np.mean(wc_list)

# =========================================================
# TAB 1: FULL SIMULATION (UNCHANGED)
# =========================================================
with tab1:

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

    df, aging_df = run_simulation(demand, reorder_point, order_qty)
    df["Date"] = pd.date_range("2024-01-01", periods=num_days)
    aging_df["Date"] = df["Date"]

    # ================= KPIs =================
    df["Blocked WC"] = df["Inventory Position"] * unit_value
    df["Inventory Value"] = df["Inventory Position"] * unit_value

    stockout_days = (df["Closing Balance"] == 0).sum()
    avg_wc = df["Blocked WC"].mean()

    st.subheader("Inventory KPIs")
    st.metric("Stockout Days", stockout_days)
    st.metric("Avg Working Capital", round(avg_wc,0))

    # ================= Inventory Chart =================
    st.subheader("Inventory Behaviour")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Closing Balance"]))

    fig.add_hline(y=reorder_point, line_dash="dash")
    fig.add_hrect(y0=0, y1=reorder_point*0.5, fillcolor="red", opacity=0.1)
    fig.add_hrect(y0=reorder_point*0.5, y1=reorder_point, fillcolor="yellow", opacity=0.1)

    st.plotly_chart(fig, use_container_width=True)

    # ================= Working Capital =================
    st.subheader("Blocked Working Capital")

    fig_wc = px.line(df, x="Date", y="Blocked WC")
    fig_wc.add_hline(y=avg_wc, line_dash="dash")

    st.plotly_chart(fig_wc, use_container_width=True)

    # ================= Aging =================
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
# TAB 2: SCENARIO ANALYSIS
# =========================================================
with tab2:

    st.header("Scenario Analysis")

    min_rp = st.number_input("Min RP", value=100)
    max_rp = st.number_input("Max RP", value=400)
    step_rp = st.number_input("RP Step", value=50)

    min_oq = st.number_input("Min OQ", value=100)
    max_oq = st.number_input("Max OQ", value=800)
    step_oq = st.number_input("OQ Step", value=100)

    if st.button("Run Scenarios"):

        results = []

        for rp in range(min_rp, max_rp+1, step_rp):
            for oq in range(min_oq, max_oq+1, step_oq):

                df_temp, _ = run_simulation(demand, rp, oq)

                avg_wc = (df_temp["Inventory Position"] * unit_value).mean()

                df_temp["Inventory Value"] = df_temp["Inventory Position"] * unit_value
                df_temp["Holding Cost"] = df_temp["Inventory Value"] * holding_cost_rate / 365

                holding_cost = df_temp["Holding Cost"].sum()
                ordering_cost_total = (df_temp["New Order"] > 0).sum() * ordering_cost

                total_cost = holding_cost + ordering_cost_total

                results.append({
                    "RP": rp,
                    "OQ": oq,
                    "Avg WC": avg_wc,
                    "Total Cost": total_cost
                })

        scenario_df = pd.DataFrame(results)

        st.dataframe(scenario_df)

        fig = px.scatter(
            scenario_df,
            x="Avg WC",
            y="Total Cost",
            hover_data=["RP", "OQ"]
        )

        st.plotly_chart(fig, use_container_width=True)
