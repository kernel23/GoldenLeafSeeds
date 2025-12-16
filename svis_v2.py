import streamlit as st
import pandas as pd
import plotly.express as px
import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from datetime import datetime
import numpy as np
import cv2
from streamlit_gsheets import GSheetsConnection

# --- PDF LIBRARIES ---
from reportlab.lib.pagesizes import landscape
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import qrcode
import io
import tempfile
import os

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

import re

def parse_smart_location(df):
    """
    Parses location strings based on 'Cabinet, Row, Column' format.
    Example: "Cabinet A, Row 2, Col 5" -> Zone='Cabinet A', Level='Row 2', Position='Col 5'
    """
    def extract_coords(loc_str):
        s = str(loc_str).strip()
        
        # 1. Primary Strategy: Comma Separation (User Preference)
        if "," in s:
            parts = [p.strip() for p in s.split(",")]
        else:
            # 2. Fallback Strategy: Hyphen Separation (e.g. "CabA-Row1")
            parts = [p.strip() for p in re.split(r'[-]+', s) if p.strip()]
        
        # Logic to map parts to [Cabinet, Row, Column]
        # We fill missing details with defaults if the user only entered "Cabinet A"
        if len(parts) == 0:
            return "Unknown", "General", "1"
        elif len(parts) == 1:
            return parts[0], "Row 1", "Col 1"      # Just Cabinet
        elif len(parts) == 2:
            return parts[0], parts[1], "Col 1"     # Cabinet + Row
        else:
            return parts[0], parts[1], parts[2]    # Cabinet + Row + Column

    # Apply
    coords = df['location'].apply(extract_coords)
    df['cabinet'] = [c[0] for c in coords]
    df['row'] = [c[1] for c in coords]
    df['column'] = [c[2] for c in coords]
    return df

# --- IMPROVED DRIVE UPLOADER ---
def upload_to_drive(file_obj, filename):
    """Uploads file to the folder ID defined in secrets."""
    try:
        # 1. Get Folder ID from Secrets
        folder_id = st.secrets.get("drive", {}).get("folder_id")
        if not folder_id:
            return False, "Folder ID not found in secrets.toml"

        # 2. Auth using existing GSheets Credentials
        creds_dict = dict(st.secrets["connections"]["gsheets"])
        creds = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=["https://www.googleapis.com/auth/drive.file"]
        )
        service = build('drive', 'v3', credentials=creds)

        # 3. Define Metadata
        file_metadata = {
            'name': f"{datetime.now().strftime('%Y-%m-%d')}_{filename}",
            'parents': [folder_id]
        }

        # 4. Upload
        media = MediaIoBaseUpload(file_obj, mimetype='text/csv', resumable=True)
        file = service.files().create(
            body=file_metadata, 
            media_body=media, 
            fields='id'
        ).execute()

        return True, file.get('id')
    except Exception as e:
        return False, str(e)
    

# --- CONFIGURATION ---
st.set_page_config(
    page_title="NTA Seed Management System", 
    layout="wide", 
    page_icon="🌱"
)

# --- 1. GOOGLE SHEETS DB MANAGER ---
def get_conn():
    return st.connection("gsheets", type=GSheetsConnection)

def load_data(worksheet):
    """Loads a specific worksheet as a DataFrame."""
    conn = get_conn()
    try:
        # ttl=0 ensures we don't cache old data; we want real-time sync
        df = conn.read(worksheet=worksheet, ttl=0)
        return df
    except Exception:
        # If sheet is empty, return structure
        return pd.DataFrame()

def save_data(df, worksheet):
    """Writes the DataFrame back to the Google Sheet."""
    conn = get_conn()
    conn.update(worksheet=worksheet, data=df)

# --- CRUD LOGIC (Refactored for DataFrames) ---

def create_batch(lot, type_, variety, qty, year, germ, loc):
    df = load_data("inventory")
    
    # Check duplicate
    if not df.empty and lot in df['lot_code'].astype(str).values:
        return False, "Error: Lot Code already exists."
    
    new_row = pd.DataFrame([{
        "lot_code": lot,
        "type": type_,
        "variety": variety,
        "quantity_g": qty,
        "year_produced": year,
        "current_germination": germ,
        "location": loc,
        "last_updated": datetime.now().strftime('%Y-%m-%d'),
        "notes": ""
    }])
    
    # Append and Save
    updated_df = pd.concat([df, new_row], ignore_index=True)
    save_data(updated_df, "inventory")
    
    # Log History
    log_test_result(lot, datetime.now().strftime('%Y-%m-%d'), germ, "Initial Entry")
    return True, "Batch synced to cloud!"

def update_batch_full(original_lot, new_lot, type_, variety, qty, year, loc):
    df = load_data("inventory")
    
    # Find index
    idx = df[df['lot_code'] == original_lot].index
    if idx.empty: return False
    
    i = idx[0]
    df.at[i, 'lot_code'] = new_lot
    df.at[i, 'type'] = type_
    df.at[i, 'variety'] = variety
    df.at[i, 'quantity_g'] = qty
    df.at[i, 'year_produced'] = year
    df.at[i, 'location'] = loc
    df.at[i, 'last_updated'] = datetime.now().strftime('%Y-%m-%d')
    
    save_data(df, "inventory")
    return True

def update_stock_quick(lot_code, amount_change, new_location=None):
    df = load_data("inventory")
    idx = df[df['lot_code'] == lot_code].index
    
    if not idx.empty:
        i = idx[0]
        if new_location:
            df.at[i, 'location'] = new_location
        else:
            # Ensure numeric
            current = float(df.at[i, 'quantity_g'])
            df.at[i, 'quantity_g'] = max(0, current + amount_change)
            
        df.at[i, 'last_updated'] = datetime.now().strftime('%Y-%m-%d')
        save_data(df, "inventory")

def log_test_result(lot_code, date, rate, notes):
    # 1. Update History Tab
    hist_df = load_data("history")
    new_log = pd.DataFrame([{
        "lot_code": lot_code, "test_date": str(date), "rate": rate, "notes": notes
    }])
    updated_hist = pd.concat([hist_df, new_log], ignore_index=True)
    save_data(updated_hist, "history")
    
    # 2. Update Main Inventory Current Germination
    inv_df = load_data("inventory")
    idx = inv_df[inv_df['lot_code'] == lot_code].index
    if not idx.empty:
        inv_df.at[idx[0], 'current_germination'] = rate
        inv_df.at[idx[0], 'last_updated'] = str(date)
        save_data(inv_df, "inventory")

def delete_batch(lot_code):
    # Delete from Inventory
    df = load_data("inventory")
    df = df[df['lot_code'] != lot_code]
    save_data(df, "inventory")
    
    # Delete from History (Optional, keeps data clean)
    hist = load_data("history")
    hist = hist[hist['lot_code'] != lot_code]
    save_data(hist, "history")

def log_daily_summary(date, stats):
    """Logs daily aggregated stats to the 'env_logs' worksheet."""
    df = load_data("env_logs")
    
    # Check if entry for this date already exists to avoid duplicates
    if not df.empty and str(date) in df['date'].astype(str).values:
        return False, "Data for this date already exists."
    
    new_row = pd.DataFrame([{
        "date": str(date),
        "avg_temp": round(stats['t_avg'], 2),
        "max_temp": round(stats['t_max'], 2),
        "min_temp": round(stats['t_min'], 2),
        "avg_humidity": round(stats['h_avg'], 2),
        "max_humidity": round(stats['h_max'], 2),
        "min_humidity": round(stats['h_min'], 2),
        "alert_count": stats['alerts']
    }])
    
    updated_df = pd.concat([df, new_row], ignore_index=True)
    save_data(updated_df, "env_logs")
    return True, "Daily summary synced to cloud!"

# --- HELPER: QR & PDF ---
def decode_qr_image(uploaded_image):
    if uploaded_image is None: return None
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(img)
    return data if data else None

def create_label_pdf(batch_data):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(4*inch, 2*inch))
    c.setLineWidth(2)
    c.rect(10, 10, 4*inch-20, 2*inch-20)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(20, 1.7*inch, "NTA TOBACCO SEED")
    c.line(20, 1.65*inch, 3.8*inch, 1.65*inch)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(20, 1.35*inch, str(batch_data['variety']))
    c.setFont("Helvetica", 9)
    c.drawString(20, 1.1*inch, f"Lot: {batch_data['lot_code']}")
    c.drawString(20, 0.9*inch, f"Germ: {batch_data['current_germination']}%")
    c.drawString(20, 0.7*inch, f"Loc: {batch_data['location']}")
    
    qr = qrcode.QRCode(box_size=10, border=1)
    qr.add_data(batch_data['lot_code'])
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        img.save(tmp.name)
        c.drawImage(tmp.name, 2.7*inch, 0.4*inch, width=1.1*inch, height=1.1*inch)
    os.unlink(tmp.name)
    c.save()
    buffer.seek(0)
    return buffer

# --- AUTHENTICATION ---
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

name, auth_status, username = authenticator.login("main")

# --- MAIN APP LOGIC ---
if auth_status:
    # --- SIDEBAR BRANDING ---
    with st.sidebar:
        # 1. Logo Handling (Graceful fallback if file missing)
        try:
            # Place 'logo.png' in your main project folder
            st.image("logo.png", use_container_width=True) 
        except Exception:
            # Fallback if no logo file found
            st.warning("⚠️ Add 'logo.png' to folder")
            st.markdown("# 🌱 NTA")

        # 2. Title & Subtitle
        st.title("NTA Seed Management System")
        st.caption("Manage and track all seed storage activities")
        
        st.divider()
        
        # 3. User Info & Logout
        st.write(f"👤 Connected: **{name}**")
        authenticator.logout('Logout', 'sidebar')
        
        st.divider()
        
        # 4. Navigation
        page = st.radio(
            "Navigation", 
            ["Dashboard", "📱 Warehouse Mode", "Receive Stock", "Analytics", "Environment"],
            label_visibility="collapsed"
        )
    # Load Data Once for Page Render
    df_inv = load_data("inventory")
    
    
    # --- 1. DASHBOARD ---
    if page == "Dashboard":
        if df_inv.empty:
            st.warning("Database is empty. Go to 'Receive Stock' to add items.")
        else:
            # --- TOP METRICS ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Batches", len(df_inv))
            col2.metric("Varieties", df_inv['variety'].nunique())
            col3.metric("Stock Weight", f"{pd.to_numeric(df_inv['quantity_g'], errors='coerce').sum()/1000:.1f} kg")

            # --- FILTERS (Global) ---
            st.divider()
            c1, c2 = st.columns([1, 2])
            filter_var = c1.multiselect("Filter Variety", df_inv['variety'].unique())
            search = c2.text_input("Search Lot Code / Notes")
            
            # Apply Filters
            show_df = df_inv.copy()
            if filter_var: show_df = show_df[show_df['variety'].isin(filter_var)]
            if search: show_df = show_df[show_df['lot_code'].astype(str).str.contains(search, case=False)]
            
            # Status Logic
            def get_status(row):
                try:
                    if float(row['current_germination']) < 80: return 'Critical'
                    if float(row['quantity_g']) < 500: return 'Low Stock'
                    return 'Good'
                except: return 'Unknown'
            
            if not show_df.empty:
                show_df['status'] = show_df.apply(get_status, axis=1)

            # ==================================================
            # SECTION 1: 🗺️ VIRTUAL WAREHOUSE
            # ==================================================
            with st.expander("🗺️ Virtual Warehouse Map", expanded=False):
                st.caption("Visual representation of stock. Format locations as: **Cabinet, Row, Column**")
                
                if show_df.empty:
                    st.info("No data to visualize.")
                else:
                    # Parse Data using the helper we defined earlier
                    map_df = parse_smart_location(show_df.copy())
                    
                    vc1, vc2 = st.columns([1, 4])
                    view_mode = vc1.radio("View Mode:", ["2D Heatmap", "3D Space View"])
                    
                    if view_mode == "2D Heatmap":
                        # Heatmap: Cabinet vs Row
                        agg = map_df.groupby(['cabinet', 'row']).agg(
                            count=('lot_code', 'count')
                        ).reset_index()
                        
                        fig_map = px.density_heatmap(
                            agg, 
                            x='cabinet', y='row', z='count',
                            text_auto=True,
                            title="Storage Density: Cabinet vs. Row",
                            color_continuous_scale='Viridis'
                        )
                        fig_map.update_xaxes(categoryorder='category ascending')
                        fig_map.update_yaxes(categoryorder='category ascending')
                        st.plotly_chart(fig_map, use_container_width=True)

                    elif view_mode == "3D Space View":
                        # 3D Scatter
                        fig_3d = px.scatter_3d(
                            map_df,
                            x='cabinet', y='row', z='column',
                            color='status',
                            symbol='type',
                            hover_data=['lot_code', 'variety', 'quantity_g'],
                            color_discrete_map={'Good': 'green', 'Low Stock': 'orange', 'Critical': 'red'},
                            title="3D Layout: Cabinet x Row x Column"
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)

            # ==================================================
            # SECTION 2: 📋 MASTER INVENTORY LIST
            # ==================================================
            with st.expander("📋 Master Inventory List", expanded=True):
                st.dataframe(
                    show_df, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "status": st.column_config.TextColumn(
                            "Health",
                            help="Status based on Germination & Qty",
                            validate="^(Good|Low Stock|Critical)$"
                        ),
                        "quantity_g": st.column_config.NumberColumn("Weight (g)"),
                        "current_germination": st.column_config.ProgressColumn("Germ %", format="%.0f%%", min_value=0, max_value=100)
                    }
                )

            # ==================================================
            # SECTION 3: 🛠️ BATCH MANAGER
            # ==================================================
            with st.expander("🛠️ Batch Manager", expanded=True):
                st.write("Select a lot below to Edit, Print Labels, or Update Records.")
                
                lot_list = show_df['lot_code'].unique()
                selected_lot = st.selectbox("Select Lot to Manage:", lot_list)
                
                if selected_lot:
                    batch = show_df[show_df['lot_code'] == selected_lot].iloc[0]
                    t1, t2, t3 = st.tabs(["🔎 Inspect & Print", "✏️ Edit Details", "🗑️ Delete"])
                    
                    # --- INSPECT TAB ---
                    with t1:
                        c1, c2 = st.columns([1,2])
                        with c1:
                            st.info(f"Germination: {batch['current_germination']}%")
                            st.write(f"Location: {batch['location']}")
                            st.write(f"Updated: {batch['last_updated']}")
                            
                            # Print Label
                            pdf = create_label_pdf(batch)
                            st.download_button("🖨️ Download Label (PDF)", pdf, file_name=f"{selected_lot}.pdf", mime="application/pdf")
                            
                            # Log Test Result
                            st.divider()
                            st.caption("Update Germination Test")
                            with st.form("test_form"):
                                d = st.date_input("Date")
                                r = st.number_input("Result %", 0, 100)
                                n = st.text_input("Notes")
                                if st.form_submit_button("Save Test Result"):
                                    log_test_result(selected_lot, d, r, n)
                                    st.success("Saved!")
                                    st.rerun()
                        with c2:
                            # Show History Chart
                            hist = load_data("history")
                            if not hist.empty:
                                hist = hist[hist['lot_code'] == selected_lot]
                                if not hist.empty:
                                    fig = px.line(hist, x='test_date', y='rate', markers=True, title="Germination History")
                                    fig.update_yaxes(range=[0, 100])
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("No test history found for this lot.")

                    # --- EDIT TAB ---
                    with t2:
                        with st.form("edit_form"):
                            c1, c2 = st.columns(2)
                            nl = c1.text_input("Lot Code", value=batch['lot_code'])
                            nv = c2.text_input("Variety", value=batch['variety'])
                            nq = c1.number_input("Qty (g)", value=float(batch['quantity_g']))
                            nloc = c2.text_input("Location", value=batch['location'], help="Format: Cabinet, Row, Column")
                            
                            if st.form_submit_button("Update Batch"):
                                update_batch_full(selected_lot, nl, batch['type'], nv, nq, batch['year_produced'], nloc)
                                st.success("Updated Successfully!")
                                st.rerun()

                    # --- DELETE TAB ---
                    with t3:
                        st.error("Danger Zone")
                        st.write("This action cannot be undone.")
                        if st.button("DELETE BATCH PERMANENTLY", type="primary"):
                            delete_batch(selected_lot)
                            st.success("Deleted.")
                            st.rerun()

    # --- 2. WAREHOUSE MODE ---
    elif page == "📱 Warehouse Mode":
        st.title("📱 Warehouse Ops")
        
        # Scan / Search
        scan_col, search_col = st.columns([1,2])
        found_lot = None
        
        with scan_col:
            img = st.camera_input("Scan", label_visibility="collapsed")
            # Inside the scan block
            if img:
                res = decode_qr_image(img)
                if res:
                    st.success(f"Scanned: {res}")
                    found_lot = res
                else:
                    st.warning("QR Code not detected. Please try moving closer or creating better lighting.")
                    
        with search_col:
            man = st.text_input("Manual Search", value=found_lot if found_lot else "")
            if man: found_lot = man
            
        if found_lot and not df_inv.empty:
            batch = df_inv[df_inv['lot_code'] == found_lot]
            if not batch.empty:
                b = batch.iloc[0]
                st.header(f"{b['variety']}")
                st.caption(f"Lot: {b['lot_code']}")
                
                m1, m2 = st.columns(2)
                m1.metric("Weight", f"{b['quantity_g']}g")
                m2.metric("Loc", f"{b['location']}")
                
                with st.expander("⚡ Quick Actions", expanded=True):
                    c_a, c_b = st.columns(2)
                    if c_a.button("➖ 50g Used"):
                        update_stock_quick(found_lot, -50)
                        st.toast("Updated Cloud!")
                        st.rerun()
                    
                    new_l = c_b.text_input("New Loc", placeholder=b['location'])
                    if c_b.button("Move"):
                        update_stock_quick(found_lot, 0, new_location=new_l)
                        st.toast("Moved!")
                        st.rerun()
            else:
                st.error("Lot not found")

    # --- 3. RECEIVE STOCK ---
    elif page == "Receive Stock":
        st.title("➕ Receive Stock")
        with st.form("new_batch"):
            c1, c2 = st.columns(2)
            l = c1.text_input("Lot Code")
            t = c2.selectbox("Type", ["Virginia", "Burley", "Oriental"])
            v = c1.text_input("Variety")
            q = c2.number_input("Qty (g)", min_value=0)
            g = c1.number_input("Germ %", 0, 100, 95)
            loc = c2.text_input("Location")
            y = st.number_input("Year", 2020, 2030, 2024)
            
            if st.form_submit_button("Receive"):
                if l:
                    success, msg = create_batch(l, t, v, q, y, g, loc)
                    if success: st.success(msg)
                    else: st.error(msg)
                else:
                    st.error("Lot Code required")

    # --- 4. ANALYTICS (DECISION SUPPORT + ENVIRONMENT) ---
    elif page == "Analytics":
        st.title("📊 Inventory Intelligence")
        st.caption("Data-driven insights for stock, quality, and environmental impact.")
        
        # Load Data
        df_inv = load_data("inventory")
        df_env_log = load_data("env_logs") 
        
        if df_inv.empty:
            st.warning("No inventory data found. Please Receive Stock first.")
        else:
            # --- 0. PRE-PROCESSING ---
            df_inv['quantity_g'] = pd.to_numeric(df_inv['quantity_g'], errors='coerce').fillna(0)
            df_inv['current_germination'] = pd.to_numeric(df_inv['current_germination'], errors='coerce').fillna(0)
            df_inv['year_produced'] = pd.to_numeric(df_inv['year_produced'], errors='coerce').fillna(0)
            
            # --- 1. KPI BOARD ---
            total_weight_kg = df_inv['quantity_g'].sum() / 1000.0
            avg_germ = df_inv['current_germination'].mean()
            low_stock_count = df_inv[df_inv['quantity_g'] < 500].shape[0]
            critical_germ_count = df_inv[df_inv['current_germination'] < 80].shape[0]

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Stock", f"{total_weight_kg:.2f} kg")
            k2.metric("Avg Germination", f"{avg_germ:.1f}%", delta=f"{avg_germ-85:.1f}% vs Target" if avg_germ < 85 else "Healthy")
            k3.metric("Low Stock Batches", low_stock_count, delta="Reorder Needed" if low_stock_count > 0 else None, delta_color="inverse")
            k4.metric("Critical Quality", critical_germ_count, delta="< 80% Germ", delta_color="inverse")
            
            st.divider()

            # --- 2. ADVANCED VISUALIZATIONS ---
            t1, t2, t3 = st.tabs(["📦 Stock Composition", "📉 Aging & Environment", "⚠️ Action Items"])
            
            with t1:
                st.subheader("Inventory Hierarchy & Health")
                st.caption("Inner: Type | Middle: Variety | Outer: Lot. (Red = Low Germination)")
                fig_sun = px.sunburst(
                    df_inv, 
                    path=['type', 'variety', 'lot_code'], 
                    values='quantity_g',
                    color='current_germination',
                    color_continuous_scale='RdYlGn',
                    range_color=[50, 100]
                )
                st.plotly_chart(fig_sun, use_container_width=True)
                
            with t2:
                # --- ENVIRONMENTAL CORRELATION ---
                st.subheader("🌍 Environment vs. Germination Trends")
                st.caption("Correlating seed production year with quality and average storage conditions.")
                
                # 1. Inventory Stats
                inv_stats = df_inv.groupby('year_produced')['current_germination'].mean().reset_index()
                inv_stats.rename(columns={'year_produced': 'Year', 'current_germination': 'Avg Germination %'}, inplace=True)
                
                # 2. Environment Stats
                if not df_env_log.empty:
                    df_env_log['date'] = pd.to_datetime(df_env_log['date'])
                    df_env_log['Year'] = df_env_log['date'].dt.year
                    env_stats = df_env_log.groupby('Year')[['avg_temp', 'avg_humidity']].mean().reset_index()
                    merged_df = pd.merge(inv_stats, env_stats, on='Year', how='outer').sort_values('Year')
                else:
                    merged_df = inv_stats
                    st.info("Note: Upload Environment Logs to see Temperature/Humidity overlays.")

                # 3. Combo Chart
                if not merged_df.empty:
                    from plotly.subplots import make_subplots
                    import plotly.graph_objects as go

                    fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_combo.add_trace(go.Bar(x=merged_df['Year'], y=merged_df['Avg Germination %'], name="Germination %", marker_color='lightgreen'), secondary_y=False)
                    
                    if 'avg_temp' in merged_df.columns:
                        fig_combo.add_trace(go.Scatter(x=merged_df['Year'], y=merged_df['avg_temp'], name="Avg Temp (°C)", mode='lines+markers', line=dict(color='red', width=3)), secondary_y=True)
                    if 'avg_humidity' in merged_df.columns:
                        fig_combo.add_trace(go.Scatter(x=merged_df['Year'], y=merged_df['avg_humidity'], name="Avg Humid (%)", mode='lines+markers', line=dict(color='blue', dash='dot')), secondary_y=True)

                    fig_combo.update_layout(title="Germination Rate & Storage Conditions per Year", legend=dict(orientation="h", y=1.1))
                    fig_combo.update_yaxes(title_text="Germination %", range=[0, 100], secondary_y=False)
                    fig_combo.update_yaxes(title_text="Temp / Humid", secondary_y=True)
                    st.plotly_chart(fig_combo, use_container_width=True)
                
                st.divider()
                
                # Standard Scatter
                st.subheader("Viability vs. Age Details")
                fig_scat = px.scatter(
                    df_inv, x='year_produced', y='current_germination', size='quantity_g', 
                    color='type', hover_data=['variety', 'lot_code'], title="Germination Scatter (Size = Qty)"
                )
                fig_scat.add_hrect(y0=0, y1=75, line_width=0, fillcolor="red", opacity=0.1, annotation_text="Critical")
                st.plotly_chart(fig_scat, use_container_width=True)

            with t3:
                st.subheader("💡 Decision Support Matrix")
                
                col_a, col_b = st.columns(2)
                
                # 1. Standard Inventory Checks
                disposal = df_inv[df_inv['current_germination'] < 75]
                try:
                    df_inv['last_updated_dt'] = pd.to_datetime(df_inv['last_updated'], errors='coerce')
                    six_months_ago = datetime.now() - pd.Timedelta(days=180)
                    stale_batches = df_inv[df_inv['last_updated_dt'] < six_months_ago]
                except:
                    stale_batches = pd.DataFrame()

                with col_a:
                    st.error(f"🚨 Low Viability ({len(disposal)})")
                    st.caption("Germination < 75%. Recommendation: Discard.")
                    if not disposal.empty:
                        st.dataframe(disposal[['lot_code', 'variety', 'current_germination', 'location']], hide_index=True)

                with col_b:
                    st.warning(f"📅 Stale Records ({len(stale_batches)})")
                    st.caption("No updates > 6 Mos. Recommendation: Retest.")
                    if not stale_batches.empty:
                        st.dataframe(stale_batches[['lot_code', 'variety', 'last_updated']], hide_index=True)

                # 2. NEW: STORAGE ENVIRONMENT ADVISORY
                st.divider()
                st.subheader("🌍 Environmental Risk Advisory")
                
                if not df_env_log.empty:
                    # Logic: Analyze last 30 days of LOGGED data
                    df_env_log['date'] = pd.to_datetime(df_env_log['date'])
                    latest_log = df_env_log['date'].max()
                    
                    # Filter last 30 days relative to the latest data
                    recent = df_env_log[df_env_log['date'] >= (latest_log - pd.Timedelta(days=30))]
                    
                    if not recent.empty:
                        # Metrics
                        r_temp = recent['avg_temp'].mean()
                        r_hum = recent['avg_humidity'].mean()
                        r_dev_t = recent['avg_temp'].std()
                        
                        # -- RULES ENGINE --
                        risks_found = False
                        
                        # Rule 1: High Temp
                        if r_temp > 25.0:
                            st.warning(f"🔥 **High Heat Detected (Avg {r_temp:.1f}°C)**")
                            st.markdown("Recent temperatures exceed 25°C. **Risk:** Accelerated seed aging. **Action:** Check Air Conditioning.")
                            risks_found = True
                            
                        # Rule 2: High Humidity
                        if r_hum > 55.0:
                            st.warning(f"💧 **High Humidity Detected (Avg {r_hum:.1f}%)**")
                            st.markdown("Recent humidity exceeds 55%. **Risk:** Mold growth. **Action:** Check Dehumidifiers / Silica.")
                            risks_found = True
                            
                        # Rule 3: Instability
                        if r_dev_t > 2.0: # High standard deviation
                            st.warning("📉 **Unstable Conditions Detected**")
                            st.markdown("Temperature is fluctuating significantly. **Risk:** Dormancy breaking. **Action:** Check Door Seals/Insulation.")
                            risks_found = True
                        
                        if risks_found:
                            st.caption("🔻 **Vulnerable Batches** (Germination 75-85%) likely to degrade first:")
                            vulnerable = df_inv[(df_inv['current_germination'] >= 75) & (df_inv['current_germination'] <= 85)]
                            if not vulnerable.empty:
                                st.dataframe(vulnerable[['lot_code', 'variety', 'current_germination', 'location']], hide_index=True)
                            else:
                                st.info("No 'On the Edge' batches found. Your stock is relatively resilient.")
                        else:
                            st.success(f"✅ Storage Environment is Stable. (Avg T: {r_temp:.1f}°C, Avg H: {r_hum:.1f}%)")
                    else:
                        st.info("Not enough recent data for environmental analysis.")
                else:
                    st.info("Upload Environment Logs to enable Environmental Decision Support.")

    # --- 5. ENVIRONMENT MONITOR ---
    elif page == "Environment":
        st.title("🌡️ Environmental Control")
        
        # 1. TABS FOR MODES
        tab_upload, tab_dashboard = st.tabs(["📥 Upload & Analyze", "📈 Cloud Dashboard"])

        # ==========================================
        # TAB 1: UPLOAD, ANALYZE & SYNC
        # ==========================================
        with tab_upload:
            st.caption("Upload a datalogger CSV to analyze it and sync to the cloud.")
            
            # A. SETTINGS
            with st.expander("⚙️ Threshold Settings", expanded=False):
                t1, t2, h1, h2 = st.columns(4)
                MIN_TEMP = t1.number_input("Min Temp (°C)", value=15.0, step=0.5)
                MAX_TEMP = t2.number_input("Max Temp (°C)", value=25.0, step=0.5)
                MIN_HUM = h1.number_input("Min Humid (%)", value=30.0, step=1.0)
                MAX_HUM = h2.number_input("Max Humid (%)", value=50.0, step=1.0)

            uploaded_file = st.file_uploader("Upload Log File", type=["csv", "xlsx"])

            if uploaded_file:
                try:
                    # --- B. LOAD & CLEAN DATA ---
                    uploaded_file.seek(0)
                    try:
                        df_env = pd.read_csv(uploaded_file)
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df_env = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

                    df_env.columns = df_env.columns.str.strip()
                    rename_map = {}
                    for col in df_env.columns:
                        if "temp" in col.lower(): rename_map[col] = "Temp"
                        elif "humid" in col.lower(): rename_map[col] = "Humid"
                        elif "time" in col.lower() or "date" in col.lower(): rename_map[col] = "Time"
                    df_env.rename(columns=rename_map, inplace=True)

                    if not {"Time", "Temp", "Humid"}.issubset(df_env.columns):
                        st.error(f"Columns missing. Found: {list(df_env.columns)}")
                        st.stop()

                    df_env['Time'] = pd.to_datetime(df_env['Time'])
                    df_env['Date'] = df_env['Time'].dt.date
                    df_env = df_env.sort_values('Time')

                    # --- C. FILTER & SEARCH (Upload Tab) ---
                    st.divider()
                    f_col1, f_col2 = st.columns([1, 3])
                    
                    with f_col1:
                        st.subheader("🔎 Filter View")
                        filter_option = st.radio(
                            "Select Range:",
                            ["All Data", "Last 8 Hours", "Last 12 Hours", "Last 24 Hours", "Specific Date", "Last 7 Days", "Specific Month", "Specific Year"]
                        )
                    
                    # Logic to Filter DataFrame
                    df_view = df_env.copy()
                    max_time = df_env['Time'].max()
                    
                    if filter_option == "Last 8 Hours":
                        df_view = df_env[df_env['Time'] >= (max_time - pd.Timedelta(hours=8))]
                    elif filter_option == "Last 12 Hours":
                        df_view = df_env[df_env['Time'] >= (max_time - pd.Timedelta(hours=12))]
                    elif filter_option == "Last 24 Hours":
                        df_view = df_env[df_env['Time'] >= (max_time - pd.Timedelta(hours=24))]
                    elif filter_option == "Last 7 Days":
                        df_view = df_env[df_env['Time'] >= (max_time - pd.Timedelta(days=7))]
                    
                    elif filter_option == "Specific Date":
                        min_d, max_d = df_env['Date'].min(), df_env['Date'].max()
                        with f_col1:
                            sel_date = st.date_input("Pick Date", value=max_d, min_value=min_d, max_value=max_d)
                        df_view = df_env[df_env['Date'] == sel_date]
                    
                    elif filter_option == "Specific Month":
                        avail_months = df_env['Time'].dt.to_period('M').unique()
                        with f_col1:
                            sel_month = st.selectbox("Choose Month", sorted(avail_months.astype(str), reverse=True))
                        df_view = df_env[df_env['Time'].dt.to_period('M').astype(str) == sel_month]
                    
                    elif filter_option == "Specific Year":
                        avail_years = df_env['Time'].dt.year.unique()
                        with f_col1:
                            sel_year = st.selectbox("Choose Year", sorted(avail_years, reverse=True))
                        df_view = df_env[df_env['Time'].dt.year == sel_year]

                    # --- D. PREVIEW CHARTS ---
                    if df_view.empty:
                        st.warning("No data found for this specific timeframe.")
                    else:
                        # Metrics
                        avg_t = df_view['Temp'].mean()
                        avg_h = df_view['Humid'].mean()
                        total_alerts = df_view[(df_view['Temp'] > MAX_TEMP) | (df_view['Temp'] < MIN_TEMP) | 
                                               (df_view['Humid'] > MAX_HUM) | (df_view['Humid'] < MIN_HUM)].shape[0]

                        with f_col2:
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Avg Temp", f"{avg_t:.1f}°C")
                            m2.metric("Avg Humid", f"{avg_h:.1f}%")
                            m3.metric("Data Points", f"{len(df_view)}")
                            m4.metric("Alerts", f"{total_alerts}", delta_color="inverse")

                        # Charts
                        tab1, tab2 = st.tabs(["Temperature", "Humidity"])
                        chart_title = filter_option if filter_option != "Specific Date" else str(sel_date)
                        
                        with tab1:
                            fig_t = px.line(df_view, x='Time', y='Temp', title=f"Temperature - {chart_title}")
                            fig_t.add_hline(y=MAX_TEMP, line_dash="dash", line_color="red")
                            fig_t.add_hline(y=MIN_TEMP, line_dash="dash", line_color="blue")
                            st.plotly_chart(fig_t, use_container_width=True)
                            
                        with tab2:
                            fig_h = px.line(df_view, x='Time', y='Humid', title=f"Humidity - {chart_title}", color_discrete_sequence=['teal'])
                            fig_h.add_hline(y=MAX_HUM, line_dash="dash", line_color="red")
                            fig_h.add_hline(y=MIN_HUM, line_dash="dash", line_color="blue")
                            st.plotly_chart(fig_h, use_container_width=True)

                    st.divider()

                    # --- E. SYNC ACTIONS ---
                    c1, c2 = st.columns(2)
                    
                    # 1. Sync Summaries
                    with c1:
                        st.subheader("1. Sync Summaries")
                        st.caption("Updates 'env_logs' with daily averages (Lightweight).")
                        if st.button("📊 Sync Daily Stats"):
                            with st.spinner("Syncing summaries..."):
                                # Use FULL df_env for syncing history, not just the view
                                daily = df_env.groupby('Date').agg(
                                    t_avg=('Temp','mean'), t_max=('Temp','max'), t_min=('Temp','min'),
                                    h_avg=('Humid','mean'), h_max=('Humid','max'), h_min=('Humid','min')
                                ).reset_index()
                                daily['alerts'] = df_env.groupby('Date').apply(
                                    lambda x: ((x['Temp']>MAX_TEMP)|(x['Temp']<MIN_TEMP)|
                                               (x['Humid']>MAX_HUM)|(x['Humid']<MIN_HUM)).sum()
                                ).values
                                
                                cloud_df = load_data("env_logs")
                                existing = set(cloud_df['date'].astype(str).values) if not cloud_df.empty else set()
                                
                                new_rows = []
                                for _, r in daily.iterrows():
                                    if str(r['Date']) not in existing:
                                        new_rows.append({
                                            "date": str(r['Date']),
                                            "avg_temp": round(r['t_avg'],2), "max_temp": round(r['t_max'],2), "min_temp": round(r['t_min'],2),
                                            "avg_humidity": round(r['h_avg'],2), "max_humidity": round(r['h_max'],2), "min_humidity": round(r['h_min'],2),
                                            "alert_count": r['alerts']
                                        })
                                if new_rows:
                                    save_data(pd.concat([cloud_df, pd.DataFrame(new_rows)], ignore_index=True), "env_logs")
                                    st.success(f"Synced {len(new_rows)} days!")
                                else:
                                    st.info("Summaries already up to date.")

                    # 2. Sync Raw Data
                    with c2:
                        st.subheader("2. Archive Raw Data")
                        st.caption("Appends to 'raw_logs' (Heavy).")
                        if st.button("💾 Sync Raw Data"):
                            with st.spinner("Archiving..."):
                                raw_cloud = load_data("raw_logs")
                                if not raw_cloud.empty and 'Time' in raw_cloud.columns:
                                    existing_times = set(pd.to_datetime(raw_cloud['Time']).astype(str))
                                    # Filter full df_env
                                    df_env['str_time'] = df_env['Time'].astype(str)
                                    new_raw = df_env[~df_env['str_time'].isin(existing_times)].drop(columns=['str_time'])
                                else:
                                    new_raw = df_env

                                if not new_raw.empty:
                                    # Ensure column order
                                    new_raw = new_raw[['Time', 'Temp', 'Humid']]
                                    updated_raw = pd.concat([raw_cloud, new_raw], ignore_index=True)
                                    save_data(updated_raw, "raw_logs")
                                    st.success(f"✅ Archived {len(new_raw)} readings!")
                                else:
                                    st.info("All data already archived.")

                except Exception as e:
                    st.error(f"Error: {e}")

        # ==========================================
        # TAB 2: CLOUD DASHBOARD (Historical)
        # ==========================================
        with tab_dashboard:
            st.header("📊 Historical Trends")
            
            df_hist = load_data("env_logs")
            
            if df_hist.empty:
                st.warning("No historical data found. Please upload files in the 'Upload' tab.")
            else:
                # Prepare Data
                df_hist['date'] = pd.to_datetime(df_hist['date'])
                df_hist = df_hist.sort_values('date')
                
                # --- NEW FILTER SECTION (Replicated) ---
                st.divider()
                f_col1, f_col2 = st.columns([1, 3])
                
                with f_col1:
                    st.subheader("🔎 Filter History")
                    hist_filter = st.radio(
                        "Select Range:",
                        ["All History", "Last 7 Days", "Last 30 Days", "Specific Month", "Specific Year", "Custom Range"],
                        key="hist_filter_radio"
                    )

                # Filter Logic
                df_view = df_hist.copy()
                max_date = df_hist['date'].max()

                if hist_filter == "Last 7 Days":
                    df_view = df_hist[df_hist['date'] >= (max_date - pd.Timedelta(days=7))]
                elif hist_filter == "Last 30 Days":
                    df_view = df_hist[df_hist['date'] >= (max_date - pd.Timedelta(days=30))]
                
                elif hist_filter == "Specific Month":
                    avail_months = df_hist['date'].dt.to_period('M').unique()
                    with f_col1:
                        sel_month = st.selectbox("Choose Month", sorted(avail_months.astype(str), reverse=True), key="hist_month")
                    df_view = df_hist[df_hist['date'].dt.to_period('M').astype(str) == sel_month]
                
                elif hist_filter == "Specific Year":
                    avail_years = df_hist['date'].dt.year.unique()
                    with f_col1:
                        sel_year = st.selectbox("Choose Year", sorted(avail_years, reverse=True), key="hist_year")
                    df_view = df_hist[df_hist['date'].dt.year == sel_year]

                elif hist_filter == "Custom Range":
                    min_d, max_d = df_hist['date'].min(), df_hist['date'].max()
                    with f_col1:
                        date_range = st.date_input("Select Range", value=(min_d, max_d), min_value=min_d, max_value=max_d, key="hist_range")
                    if isinstance(date_range, tuple) and len(date_range) == 2:
                        start_date, end_date = date_range
                        mask = (df_hist['date'].dt.date >= start_date) & (df_hist['date'].dt.date <= end_date)
                        df_view = df_hist.loc[mask]

                # --- METRICS & VISUALS (Filtered) ---
                if not df_view.empty:
                    with f_col2:
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Days Monitored", len(df_view))
                        m2.metric("Avg Temp", f"{df_view['avg_temp'].mean():.1f}°C")
                        m3.metric("Max Peak", f"{df_view['max_temp'].max():.1f}°C")
                        m4.metric("Total Alerts", int(df_view['alert_count'].sum()), delta_color="inverse")

                    # Charts
                    st.divider()
                    tab_t, tab_h = st.tabs(["Temperature Trends", "Humidity Trends"])
                    
                    with tab_t:
                        fig_t = px.line(df_view, x='date', y=['max_temp', 'avg_temp', 'min_temp'], 
                                        title=f"Temperature History - {hist_filter}",
                                        color_discrete_map={'max_temp': 'red', 'avg_temp': 'orange', 'min_temp': 'blue'})
                        st.plotly_chart(fig_t, use_container_width=True)
                        
                    with tab_h:
                        fig_h = px.line(df_view, x='date', y=['max_humidity', 'avg_humidity', 'min_humidity'], 
                                        title=f"Humidity History - {hist_filter}",
                                        color_discrete_map={'max_humidity': 'teal', 'avg_humidity': 'green', 'min_humidity': 'lightblue'})
                        st.plotly_chart(fig_h, use_container_width=True)
                    
                    # --- NEW: CORRELATION ANALYSIS ---
                    st.divider()
                    st.subheader("🔗 Correlation Analysis")
                    st.caption("Understanding relationships between Temperature, Humidity, and Alerts.")
                    
                    if len(df_view) > 2:
                        # Calculation
                        corr = df_view['avg_temp'].corr(df_view['avg_humidity'])
                        
                        cor_c1, cor_c2 = st.columns([1, 3])
                        
                        with cor_c1:
                            st.metric("Correlation Coefficient", f"{corr:.2f}")
                            
                            if abs(corr) > 0.7:
                                st.info("Strong Relationship")
                                st.markdown("""
                                **Insight:** There is a significant link. 
                                If negative (-), cooling the room drastically increases humidity.
                                """)
                            elif abs(corr) > 0.3:
                                st.info("Moderate Relationship")
                            else:
                                st.info("Weak/No Relationship")
                                st.caption("Variables appear independent.")
                                
                        with cor_c2:
                            # Scatter Plot: Temp vs Humid
                            fig_corr = px.scatter(
                                df_view, 
                                x='avg_temp', 
                                y='avg_humidity', 
                                color='alert_count',
                                size='max_temp',
                                hover_data=['date'],
                                title=f"Temp vs Humidity Scatter (Color = Alerts)",
                                color_continuous_scale='Turbo'
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.info("Not enough data points selected for correlation analysis (Need > 2 days).")


                    # Deep Dive Section
                    st.divider()
                    with st.expander("🔎 Deep Dive: Inspect Single Day (Raw Data)"):
                        st.caption("Pull granular data from 'raw_logs' sheet.")
                        selected_inspect_date = st.date_input("Select Date", value=df_hist['date'].max(), key="inspect_date")
                        
                        if st.button("Load Raw Data", key="load_raw_btn"):
                            with st.spinner("Fetching..."):
                                raw_df = load_data("raw_logs")
                                if not raw_df.empty:
                                    raw_df['Time'] = pd.to_datetime(raw_df['Time'])
                                    day_data = raw_df[raw_df['Time'].dt.date == selected_inspect_date]
                                    
                                    if not day_data.empty:
                                        st.success(f"Loaded {len(day_data)} readings")
                                        fig_detailed = px.line(day_data, x='Time', y=['Temp', 'Humid'], 
                                                               title=f"Intra-day Detail: {selected_inspect_date}")
                                        st.plotly_chart(fig_detailed, use_container_width=True)
                                        st.dataframe(day_data, use_container_width=True)
                                    else:
                                        st.warning("No raw data found for this date.")
                                else:
                                    st.error("Raw Logs empty.")
elif auth_status is False:
    st.error('Login Failed')
elif auth_status is None:
    st.warning('Please Login')