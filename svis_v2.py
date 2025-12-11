import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="GreenLeaf Manager", layout="wide", page_icon="🌱")
DB_FILE = "inventory_v2.db"

# --- DATABASE MANAGER ---
def get_conn():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON") # Enable foreign key support
    return conn

# READ (All Inventory)
def load_inventory():
    conn = get_conn()
    try:
        df = pd.read_sql("SELECT * FROM inventory", conn)
        # Calculate Logic Status
        def get_status(row):
            if row['current_germination'] < 80: return 'Critical'
            if row['quantity_g'] < 500: return 'Low Stock'
            return 'Good'
        if not df.empty:
            df['status'] = df.apply(get_status, axis=1)
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()

# READ (History Logs)
def load_history(inventory_id):
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM germination_logs WHERE inventory_id = ? ORDER BY test_date DESC", conn, params=(inventory_id,))
    conn.close()
    return df

# CREATE
def create_batch(lot, type_, variety, qty, year, germ, loc):
    conn = get_conn()
    cursor = conn.cursor()
    try:
        # 1. Insert into Inventory
        cursor.execute("""
            INSERT INTO inventory (lot_code, type, variety, quantity_g, year_produced, current_germination, location, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (lot, type_, variety, qty, year, germ, loc, datetime.now().strftime('%Y-%m-%d')))
        
        # 2. Initialize History Log
        new_id = cursor.lastrowid
        cursor.execute("INSERT INTO germination_logs (inventory_id, test_date, rate, notes) VALUES (?, ?, ?, ?)",
                       (new_id, datetime.now().strftime('%Y-%m-%d'), germ, "Initial Entry"))
        conn.commit()
        return True, "Batch created successfully!"
    except sqlite3.IntegrityError:
        return False, "Error: Lot Code might already exist."
    finally:
        conn.close()

# UPDATE (Edit Details)
def update_batch_details(id_, lot, type_, variety, qty, year, loc):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE inventory 
        SET lot_code=?, type=?, variety=?, quantity_g=?, year_produced=?, location=?, last_updated=?
        WHERE id=?
    """, (lot, type_, variety, qty, year, loc, datetime.now().strftime('%Y-%m-%d'), id_))
    conn.commit()
    conn.close()

# UPDATE (Add Test Result)
def add_test_result(inv_id, date, rate, notes):
    conn = get_conn()
    cursor = conn.cursor()
    # 1. Add Log
    cursor.execute("INSERT INTO germination_logs (inventory_id, test_date, rate, notes) VALUES (?, ?, ?, ?)", 
                   (inv_id, date, rate, notes))
    # 2. Update Current State
    cursor.execute("UPDATE inventory SET current_germination = ?, last_updated = ? WHERE id = ?", 
                   (rate, date, inv_id))
    conn.commit()
    conn.close()

# DELETE
def delete_batch(inv_id):
    conn = get_conn()
    cursor = conn.cursor()
    # Logs delete automatically if ON DELETE CASCADE is set, but we do it manually to be safe
    cursor.execute("DELETE FROM germination_logs WHERE inventory_id = ?", (inv_id,))
    cursor.execute("DELETE FROM inventory WHERE id = ?", (inv_id,))
    conn.commit()
    conn.close()

# --- UI START ---

# Sidebar Navigation
with st.sidebar:
    st.title("🌱 GreenLeaf CRUD")
    page = st.radio("Navigation", ["Dashboard", "Receive New Batch", "Analytics"])
    st.markdown("---")

# 1. DASHBOARD (READ + UPDATE + DELETE)
if page == "Dashboard":
    df = load_inventory()
    
    if df.empty:
        st.warning("Inventory is empty. Go to 'Receive New Batch' to start.")
    else:
        # --- FILTERS ---
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filter_var = st.multiselect("Filter Variety", df['variety'].unique())
        with col_f2:
            search_txt = st.text_input("Search Lot Code")
        
        filtered_df = df.copy()
        if filter_var: filtered_df = filtered_df[filtered_df['variety'].isin(filter_var)]
        if search_txt: filtered_df = filtered_df[filtered_df['lot_code'].str.contains(search_txt, case=False)]

        # --- MAIN GRID ---
        st.subheader("📦 Inventory List")
        st.dataframe(
            filtered_df[['lot_code', 'variety', 'quantity_g', 'current_germination', 'status', 'location']], 
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        
        # --- BATCH MANAGER (The CRUD Center) ---
        st.subheader("🛠️ Batch Manager")
        
        # Select Batch
        batch_map = filtered_df.set_index('id')['lot_code'].to_dict()
        selected_id = st.selectbox("Select Batch to Manage:", options=batch_map.keys(), format_func=lambda x: batch_map[x])
        
        if selected_id:
            batch = df[df['id'] == selected_id].iloc[0]
            
            # Create Tabs for different CRUD actions
            tab1, tab2, tab3 = st.tabs(["🔎 Inspect & History", "✏️ Edit Details", "🗑️ Danger Zone"])
            
            # TAB 1: INSPECT (READ HISTORY + ADD TEST)
            with tab1:
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.info(f"**Current Germination: {batch['current_germination']}%**")
                    st.write(f"**Location:** {batch['location']}")
                    st.write(f"**Weight:** {batch['quantity_g']}g")
                    
                    with st.expander("🧪 Record New Test"):
                        with st.form("add_test"):
                            t_date = st.date_input("Date", datetime.now())
                            t_rate = st.number_input("Result (%)", min_value=0, max_value=100, value=int(batch['current_germination']))
                            t_notes = st.text_input("Notes")
                            if st.form_submit_button("Save Test"):
                                add_test_result(selected_id, t_date, t_rate, t_notes)
                                st.success("Test recorded!")
                                st.rerun()

                with c2:
                    history = load_history(selected_id)
                    if not history.empty:
                        fig = px.line(history, x='test_date', y='rate', markers=True, title="Germination History")
                        fig.update_yaxes(range=[0, 105])
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(history[['test_date', 'rate', 'notes']], hide_index=True)

            # TAB 2: UPDATE (CORRECTION)
            with tab2:
                st.write("Update batch details (e.g., correct a typo or move location).")
                with st.form("edit_details"):
                    e_lot = st.text_input("Lot Code", value=batch['lot_code'])
                    c_a, c_b = st.columns(2)
                    e_type = c_a.text_input("Type", value=batch['type'])
                    e_var = c_b.text_input("Variety", value=batch['variety'])
                    
                    c_c, c_d = st.columns(2)
                    e_qty = c_c.number_input("Quantity (g)", value=int(batch['quantity_g']))
                    e_year = c_d.number_input("Year", value=int(batch['year_produced']))
                    
                    e_loc = st.text_input("Location", value=batch['location'])
                    
                    if st.form_submit_button("Update Batch"):
                        update_batch_details(selected_id, e_lot, e_type, e_var, e_qty, e_year, e_loc)
                        st.success("Batch details updated.")
                        st.rerun()

            # TAB 3: DELETE
            with tab3:
                st.error("⚠️ Danger Zone: Deleting this batch will remove it and all its history permanently.")
                if st.button(f"DELETE {batch['lot_code']}", type="primary"):
                    delete_batch(selected_id)
                    st.toast(f"Batch {batch['lot_code']} deleted.")
                    st.rerun()

# 2. CREATE PAGE
elif page == "Receive New Batch":
    st.title("➕ Receive New Inventory")
    st.write("Enter details for a new seed batch entering the warehouse.")
    
    with st.form("create_batch"):
        col1, col2 = st.columns(2)
        new_lot = col1.text_input("Lot Code (Required)")
        new_type = col2.selectbox("Type", ["Virginia", "Burley", "Oriental", "Dark", "Other"])
        
        col3, col4 = st.columns(2)
        new_var = col3.text_input("Variety Name")
        new_year = col4.number_input("Year Produced", min_value=2000, max_value=2030, value=datetime.now().year)
        
        col5, col6 = st.columns(2)
        new_qty = col5.number_input("Initial Weight (g)", min_value=0)
        new_germ = col6.number_input("Initial Germination (%)", min_value=0, max_value=100, value=95)
        
        new_loc = st.text_input("Storage Location")
        
        submitted = st.form_submit_button("Create Batch")
        
        if submitted:
            if new_lot:
                success, msg = create_batch(new_lot, new_type, new_var, new_qty, new_year, new_germ, new_loc)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                st.error("Lot Code is required.")

# 3. ANALYTICS PAGE
elif page == "Analytics":
    df = load_inventory()
    st.title("📊 Analytics")
    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Quantity by Variety")
            fig_qty = px.pie(df, names='variety', values='quantity_g', hole=0.4)
            st.plotly_chart(fig_qty, use_container_width=True)
        with col2:
            st.subheader("Average Viability by Year")
            avg_germ = df.groupby('year_produced')['current_germination'].mean().reset_index()
            fig_germ = px.bar(avg_germ, x='year_produced', y='current_germination')
            st.plotly_chart(fig_germ, use_container_width=True)