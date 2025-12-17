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
import base64

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

import re
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch, mm
from reportlab.graphics.shapes import Drawing, Group
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics import renderPDF

def create_flexible_label(batch_data, layout_type="Thermal 4x2"):
    """
    Generates a PDF label with corrected A4 dimensions to prevent cutting.
    """
    filename = f"label_{batch_data['lot_code']}.pdf"
    
    # === OPTION 1: THERMAL PRINTER (4x2 inches) ===
    if layout_type == "Thermal 4x2":
        # (This part was already working fine, keeping it same)
        c = canvas.Canvas(filename, pagesize=(4*inch, 2*inch))
        
        # QR Code Setup
        qr_size = 1.4 * inch
        qr = QrCodeWidget(batch_data['lot_code'])
        bounds = qr.getBounds()
        scale = qr_size / (bounds[2] - bounds[0])
        
        d = Drawing(qr_size, qr_size)
        d.add(Group(qr, transform=[scale, 0, 0, scale, 0, 0]))
        renderPDF.draw(d, c, 0.2*inch, 0.3*inch) 
        
        # Text Setup
        text_x = 1.8 * inch 
        c.setFont("Helvetica-Bold", 16)
        c.drawString(text_x, 1.6*inch, f"{batch_data['variety']}")
        c.setFont("Helvetica", 11)
        c.drawString(text_x, 1.35*inch, f"Lot: {batch_data['lot_code']}")
        c.drawString(text_x, 1.15*inch, f"Type: {batch_data['type']}")
        c.setFont("Helvetica-Bold", 11)
        c.drawString(text_x, 0.95*inch, f"Germ: {batch_data['current_germination']}%")
        c.drawString(text_x, 0.75*inch, f"Qty: {float(batch_data['quantity_g']):,.0f}g")
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(text_x, 0.3*inch, f"Loc: {batch_data['location']}")
        
        c.save()
        return filename

    # === OPTION 2: A4 SHEET (Standard Avery 3x8 Layout) ===
    elif layout_type == "A4 Sheet":
        c = canvas.Canvas(filename, pagesize=A4)
        width, height = A4 # A4 is 210mm x 297mm
        
        # --- CORRECTED DIMENSIONS FOR A4 ---
        rows = 8
        cols = 3
        
        # Standard Margins for Avery L7160
        margin_x = 7.0 * mm  # Left/Right margin
        margin_y = 13.0 * mm # Top/Bottom margin
        
        # Precise sticker size
        col_width = 63.5 * mm 
        row_height = 38.1 * mm 
        
        for r in range(rows):
            for col in range(cols):
                # Calculate Coordinates
                x = margin_x + (col * col_width)
                # Calculate Y from top down
                y = height - margin_y - ((r + 1) * row_height)
                
                # --- DRAWING CONTENT ---
                # 1. QR Code (20mm size fits well in this smaller height)
                qr_size = 20 * mm
                qr = QrCodeWidget(batch_data['lot_code'])
                bounds = qr.getBounds()
                scale = qr_size / (bounds[2] - bounds[0])
                
                d = Drawing(qr_size, qr_size)
                d.add(Group(qr, transform=[scale, 0, 0, scale, 0, 0]))
                
                # Draw QR (offset slightly from left edge of sticker)
                renderPDF.draw(d, c, x + 2*mm, y + 9*mm)
                
                # 2. Text Content (To the right of QR)
                tx = x + 24 * mm # Start text after QR
                
                # Variety Name (Truncate if too long to avoid overrun)
                var_name = batch_data['variety']
                if len(var_name) > 15: var_name = var_name[:15] + "..."
                
                c.setFont("Helvetica-Bold", 10)
                c.drawString(tx, y + 26*mm, var_name)
                
                c.setFont("Helvetica", 8)
                c.drawString(tx, y + 21*mm, f"Lot: {batch_data['lot_code']}")
                c.drawString(tx, y + 17*mm, f"Germ: {batch_data['current_germination']}%")
                c.drawString(tx, y + 13*mm, f"Qty: {float(batch_data['quantity_g']):.0f}g")
                
                # Location at bottom
                c.setFont("Helvetica-Oblique", 7)
                c.drawString(tx, y + 8*mm, f"{batch_data['location']}")
                
                # 3. Cut Guides (Optional: Very light gray box to help you see edges)
                # Set to transparent or very light gray. 
                # Good for testing, remove if printing on pre-cut stickers.
                c.setStrokeColorRGB(0.9, 0.9, 0.9) 
                c.setLineWidth(0.5)
                c.rect(x, y, col_width, row_height)
                
        c.save()
        return filename
    
    return None


def apply_mobile_styles():
    st.markdown("""
        <style>
            /* 1. Reduce top padding (Streamlit wastes a lot of space at the top) */
            .block-container {
                padding-top: 1rem !important;
                padding-bottom: 5rem !important; /* Space for scrolling on phone */
            }
            
            /* 2. Make Input Fields Taller (Easier to tap) */
            input, select, textarea {
                font-size: 16px !important;
                padding: 12px !important;
                border-radius: 8px !important;
            }
            
            /* 3. Make Buttons Big and Touchable */
            button {
                height: auto !important;
                padding-top: 10px !important;
                padding-bottom: 10px !important;
            }
            
            /* 4. Improve Tab Styling for Touch */
            button[data-baseweb="tab"] {
                font-size: 16px !important;
                padding: 10px 20px !important;
                flex: 1 1 0px; /* Make tabs expand to fill width */
            }
            
            /* 5. Hide the little "Manage App" button on mobile if it gets in the way */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

# CALL THIS FUNCTION RIGHT AFTER st.set_page_config()
apply_mobile_styles()

def get_img_as_base64(file_path):
    """Reads an image file and converts it to a base64 string for HTML embedding."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


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


def log_transaction(lot_code, action, qty, reason, notes, user="Unknown"):
    """
    Logs seed usage, disposal, or additions to the 'transactions' sheet.
    action: 'IN' (Add) or 'OUT' (Remove)
    """
    try:
        # 1. Load Data
        df_inv = load_data("inventory")
        df_trans = load_data("transactions")
        
        # 2. Get Variety Info (for easier reporting later)
        variety = "Unknown"
        if not df_inv.empty:
            match = df_inv[df_inv['lot_code'] == lot_code]
            if not match.empty:
                variety = match.iloc[0]['variety']

        # 3. Create Record
        new_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lot_code": str(lot_code),
            "variety": variety,
            "action": action,     # "IN" or "OUT"
            "quantity_g": float(qty),
            "reason": reason,
            "user": user,
            "notes": notes
        }
        
        # 4. Save
        updated_trans = pd.concat([df_trans, pd.DataFrame([new_record])], ignore_index=True)
        save_data(updated_trans, "transactions")
        return True
    except Exception as e:
        print(f"Log Error: {e}")
        return False
    
# --- HELPER FUNCTIONS ---

def update_batch_qty(lot_code, change_amount):
    """
    Updates the quantity of a specific batch in the 'inventory' sheet.
    Positive change_amount = Add. Negative change_amount = Remove.
    """
    df = load_data("inventory")
    
    # Check if the lot exists
    if lot_code in df['lot_code'].values:
        # Find the specific row index
        idx = df.index[df['lot_code'] == lot_code][0]
        
        # Calculate new quantity
        current = float(df.at[idx, 'quantity_g'])
        new_qty = current + change_amount
        
        # Update values (Ensure qty never goes below 0)
        df.at[idx, 'quantity_g'] = max(0, new_qty)
        df.at[idx, 'last_updated'] = datetime.now().strftime("%Y-%m-%d")
        
        # Save back to Google Sheets
        save_data(df, "inventory")
        return True
    
    return False

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

# --- AUTHENTICATION & LOGIN LOGIC ---
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("⚠️ Error: 'config.yaml' not found.")
    st.stop()

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Initialize session state
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None

# --- LOGIC BRANCHING ---
# 1. IF NOT LOGGED IN: Render the Modern Login Page
if st.session_state['authentication_status'] is not True:
    
    # A. Inject Custom CSS (Simplified now that image is handled via HTML)
    st.markdown(
        """
        <style>
            /* Hide Sidebar */
            [data-testid="stSidebar"] { display: none; }
            
            /* Center the Main Container vertically */
            .main .block-container {
                display: flex;
                justify-content: center;
                align-items: center;
                padding-top: 3rem !important;
            }
            
            /* Style the Login Card */
            div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stForm"]) {
                background-color: #ffffff;
                padding: 3rem;
                border-radius: 20px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                width: 100%;
                text-align: center; /* Centers text */
                border: 1px solid #e0e0e0;
            }

            div[data-testid="stFormSubmitButton"] {
                display: flex !important;
                justify-content: center !important;
                width: 100% !important;
            }

            /* Input & Button Styling */
            input { border-radius: 10px !important; padding: 10px !important; }
            button[kind="primaryFormSubmit"] {
                background: linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%) !important;
                border: none; border-radius: 10px; padding: 12px; width: 100%; color: white !important;
                font-weight: bold; margin: 0 auto !important;
            
            }
            
            /* Typography */
            h1 { color: #1B5E20; font-weight: 800; margin-bottom: 0.5rem; text-align: center; }
            p { text-align: center; color: #666; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # B. Create Columns to Center Content horizontally
    col1, col2, col3 = st.columns([1, 2, 1])

    # C. Render Content INSIDE the Center Column
    with col2:
        # 1. Logo (Centered via HTML wrapper)
        img_b64 = get_img_as_base64("logo.png")
        if img_b64:
            st.markdown(
                f"""
                <div style="text-align: center; margin-bottom: 20px;">
                    <img src="data:image/png;base64,{img_b64}" width="180">
                </div>
                """,
                unsafe_allow_html=True
            )

        # 2. Title & Subtitle
        st.markdown("<h1>NTA Seed Management System</h1>", unsafe_allow_html=True)
        st.markdown("<p style='margin-bottom:2rem;'>Manage and track all seed storage activities</p>", unsafe_allow_html=True)

        # 3. RENDER THE LOGIN FORM
        name, auth_status, username = authenticator.login("main")

        # 4. Error Messages ONLY
        if auth_status is False:
            st.error('❌ Incorrect username or password')

# 2. IF LOGGED IN: Render the Main App
else:
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
        
        # 4. Navigation
        page = st.radio(
            "Navigation", 
            ["Dashboard", "📱 Warehouse Mode", "Add Seed Stock", "Analytics", "Environment"],
            label_visibility="collapsed"
        )
        st.divider()
        # 3. User Info & Logout
        st.write(f"👤 Connected: **{name}**")
        authenticator.logout('Logout', 'sidebar')
        st.divider()
    # Load Data Once for Page Render
    df_inv = load_data("inventory")
    
    
    # --- 1. DASHBOARD ---
    if page == "Dashboard":
        if df_inv.empty:
            st.warning("Database is empty. Go to 'Receive Stock' to add items.")
        else:
            # --- TOP METRICS ---
            # Using responsive columns for mobile
            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)
            
            with m1: st.metric("Total Batches", len(df_inv))
            with m2: st.metric("Varieties", df_inv['variety'].nunique())
            with m3: st.metric("Stock Weight", f"{pd.to_numeric(df_inv['quantity_g'], errors='coerce').sum()/1000:.1f} kg")
            with m4: 
                low_stock = len(df_inv[pd.to_numeric(df_inv['quantity_g'], errors='coerce') < 500])
                st.metric("Low Stock", low_stock, delta_color="inverse")

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
                    # Parse Data using the helper
                    map_df = parse_smart_location(show_df.copy())
                    
                    vc1, vc2 = st.columns([1, 4])
                    view_mode_map = vc1.radio("Map Mode:", ["2D Heatmap", "3D Space View"])
                    
                    if view_mode_map == "2D Heatmap":
                        # Heatmap: Cabinet vs Row
                        agg = map_df.groupby(['cabinet', 'row']).agg(count=('lot_code', 'count')).reset_index()
                        fig_map = px.density_heatmap(
                            agg, x='cabinet', y='row', z='count', text_auto=True,
                            title="Storage Density: Cabinet vs. Row", color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig_map, use_container_width=True)

                    elif view_mode_map == "3D Space View":
                        # 3D Scatter
                        fig_3d = px.scatter_3d(
                            map_df, x='cabinet', y='row', z='column', color='status', symbol='type',
                            hover_data=['lot_code', 'variety', 'quantity_g'],
                            color_discrete_map={'Good': 'green', 'Low Stock': 'orange', 'Critical': 'red'},
                            title="3D Layout: Cabinet x Row x Column"
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)

            # ==================================================
            # SECTION 2: 📋 MASTER INVENTORY (UPDATED WITH CARD VIEW)
            # ==================================================
            st.write("### 📦 Inventory List")
            
            # Toggle between List (Desktop) and Cards (Mobile)
            view_toggle = st.radio("View Format:", ["📄 Table View", "📇 Card View (Mobile)"], horizontal=True, label_visibility="collapsed")
            
            if view_toggle == "📄 Table View":
                st.dataframe(
                    show_df, 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "status": st.column_config.TextColumn("Health", validate="^(Good|Low Stock|Critical)$"),
                        "quantity_g": st.column_config.NumberColumn("Weight (g)", format="%.1f g"),
                        "current_germination": st.column_config.ProgressColumn("Germ %", format="%.0f%%", min_value=0, max_value=100)
                    }
                )
            
            else:
                # --- CARD VIEW LOGIC ---
                # Pagination Setup
                items_per_page = 10
                if 'page_num' not in st.session_state: st.session_state.page_num = 0
                
                total_items = len(show_df)
                start_idx = st.session_state.page_num * items_per_page
                end_idx = start_idx + items_per_page
                batch_view = show_df.iloc[start_idx:end_idx]

                st.caption(f"Showing {start_idx+1}-{min(end_idx, total_items)} of {total_items} items")

                # Responsive Grid (2 columns)
                grid_cols = st.columns(2)
                
                for index, row in batch_view.iterrows():
                    with grid_cols[index % 2]:
                        with st.container(border=True):
                            # Header
                            st.markdown(f"#### 🌱 {row['variety']}")
                            st.caption(f"Lot: **{row['lot_code']}** | Type: {row['type']}")
                            
                            # Metrics
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("**Weight:**")
                                st.markdown(f":green[{float(row['quantity_g']):,.1f} g]")
                            with c2:
                                germ = float(row['current_germination'])
                                st.markdown("**Germination:**")
                                st.markdown(f":{'red' if germ < 80 else 'green'}[{germ}%]")
                            
                            st.caption(f"📍 {row['location']}")
                            
                            # Quick Status Badge
                            if row['status'] == 'Critical':
                                st.error("⚠️ Low Viability")
                            elif row['status'] == 'Low Stock':
                                st.warning("📉 Low Stock")

                # Pagination Controls
                st.divider()
                c_prev, c_curr, c_next = st.columns([1, 2, 1])
                with c_prev:
                    if st.session_state.page_num > 0:
                        if st.button("⬅️ Prev", use_container_width=True):
                            st.session_state.page_num -= 1
                            st.rerun()
                with c_next:
                    if end_idx < total_items:
                        if st.button("Next ➡️", use_container_width=True):
                            st.session_state.page_num += 1
                            st.rerun()

            # ==================================================
            # SECTION 3: 🛠️ BATCH MANAGER
            # ==================================================
            with st.expander("🛠️ Batch Manager", expanded=False):
                st.write("Select a lot below to Edit, Print Labels, or Update Records.")
                
                lot_list = show_df['lot_code'].unique()
                selected_lot = st.selectbox("Select Lot to Manage:", lot_list)
                
                if selected_lot:
                    batch = show_df[show_df['lot_code'] == selected_lot].iloc[0]
                    t1, t2, t3 = st.tabs(["🔎 Inspect & Print", "✏️ Edit Details", "🗑️ Delete"])
                    
                    # --- INSPECT TAB ---
                    with t1:
                        c1, c2 = st.columns([1,2])
                        # --- INSPECT TAB ---
                    with t1:
                        c1, c2 = st.columns([1,2])
                        with c1:
                            st.info(f"Germination: {batch['current_germination']}%")
                            st.write(f"Location: {batch['location']}")
                            
                            st.divider()
                            st.write("🖨️ **Label Center**")
                            
                            # SELECT PRINTER TYPE
                            printer_type = st.radio("Paper Type:", ["Thermal Roll (4x2)", "A4 Sticker Sheet"], horizontal=True)
                            layout_code = "Thermal 4x2" if printer_type == "Thermal Roll (4x2)" else "A4 Sheet"
                            
                            # GENERATE
                            if st.button("Generate Label PDF"):
                                pdf_file = create_flexible_label(batch, layout_type=layout_code)
                                
                                # Read file for download
                                with open(pdf_file, "rb") as f:
                                    pdf_bytes = f.read()
                                    
                                st.download_button(
                                    label="⬇️ Download PDF",
                                    data=pdf_bytes,
                                    file_name=f"Label_{batch['lot_code']}.pdf",
                                    mime="application/pdf"
                                )
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


    # --- 2. WAREHOUSE MODE (Scanner & Transactions) ---
    elif page == "📱 Warehouse Mode":
        st.title("📱 Warehouse Terminal")
        st.caption("Scan QR, Adjust Stock, or Move Locations.")

        # Load Data
        df_inv = load_data("inventory")

        if df_inv.empty:
            st.warning("Inventory is empty.")
            st.stop()

        # --- A. DUAL INPUT: CAMERA OR DROPDOWN ---
        # 1. Camera Input (Restored)
        # We put this in an expander so it doesn't take up space if not needed
        with st.expander("📷 Open Camera Scanner", expanded=False):
            img_file = st.camera_input("Take a picture of the QR Code")
        
        # Logic to decode QR if image is taken
        scanned_code = None
        if img_file is not None:
            # Convert the file to an OpenCV image
            bytes_data = img_file.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Detect and Decode
            detector = cv2.QRCodeDetector()
            data, bbox, _ = detector.detectAndDecode(cv_img)
            
            if data:
                scanned_code = data
                st.success(f"✅ Scanned: {scanned_code}")
            else:
                st.warning("Could not read QR code. Try moving closer.")

        # 2. Manual Search / Select
        # If we scanned something, we force the selectbox to match it
        lot_list = df_inv['lot_code'].unique()
        
        # Determine index
        default_ix = 0
        
        # Priority 1: Just Scanned Code
        if scanned_code and scanned_code in lot_list:
            default_ix = list(lot_list).index(scanned_code)
            st.session_state['last_selected_lot'] = scanned_code
            
        # Priority 2: Previously Selected in Session
        elif 'last_selected_lot' in st.session_state and st.session_state['last_selected_lot'] in lot_list:
            default_ix = list(lot_list).index(st.session_state['last_selected_lot'])

        selected_lot = st.selectbox("🔍 Select or Scan Batch:", lot_list, index=default_ix)
        
        # Save selection to session state so it persists across button clicks
        if selected_lot != st.session_state.get('last_selected_lot'):
            st.session_state['last_selected_lot'] = selected_lot
            st.rerun()

        # --- B. BATCH OPERATIONS ---
        if selected_lot:
            # Get Batch Details
            batch = df_inv[df_inv['lot_code'] == selected_lot].iloc[0]
            current_qty = float(batch['quantity_g'])
            
            st.divider()

            # Info Card
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader(f"🏷️ {batch['variety']}")
                st.caption(f"Lot: {batch['lot_code']} | Type: {batch['type']}")
                st.info(f"📍 Current Location: **{batch['location']}**")
            with c2:
                st.metric("Stock Level", f"{current_qty:,.1f} g")
                germ = float(batch['current_germination'])
                st.metric("Germination", f"{germ}%", delta="Critical" if germ < 80 else "Good", delta_color="inverse" if germ < 80 else "normal")

            st.divider()

            # --- C. TRANSACTION TABS ---
            st.write("### 📦 Actions")
            
            # Added "Move Stock" tab
            tab_out, tab_in, tab_move = st.tabs(["📤 Remove (Usage)", "📥 Add (Return)", "📍 Move Location"])

           # === TAB 1: REMOVE STOCK ===
            with tab_out:
                with st.form("remove_stock_form"):
                    c_qty, c_reason = st.columns(2)
                    remove_qty = c_qty.number_input("Amount to Remove (g)", 0.1, max_value=current_qty, value=100.0, step=10.0)
                    reason_out = c_reason.selectbox("Reason", ["Planting", "Germination Test", "Disposal", "Transfer Out"])
                    
                    # --- ADDED THIS BACK ---
                    notes_out = st.text_input("Notes (Optional)", placeholder="e.g. Given to Farmer John")
                    # -----------------------

                    st.write(f"📉 New Balance: :red[{current_qty - remove_qty:,.1f} g]")
                    
                    if st.form_submit_button("Confirm Removal", type="primary", use_container_width=True):
                        # 1. Update Inventory Qty
                        update_batch_qty(selected_lot, -remove_qty)
                        
                        # 2. Log Transaction (Now 'notes_out' exists!)
                        current_user = st.session_state.get('name', 'Admin')
                        # Ensure you have the log_transaction function defined at the top of your script
                        log_transaction(selected_lot, "OUT", remove_qty, reason_out, notes_out, current_user)
                        
                        st.success(f"✅ Removed {remove_qty}g & Logged!")
                        st.rerun()

            # === TAB 2: ADD STOCK ===
            with tab_in:
                with st.form("add_stock_form"):
                    c_qty, c_reason = st.columns(2)
                    add_qty = c_qty.number_input("Amount to Add (g)", 0.1, value=500.0, step=50.0)
                    reason_in = c_reason.selectbox("Reason", ["New Harvest", "Return Stock", "Inventory Adjustment"])
                    
                    # --- ADDED THIS BACK ---
                    notes_in = st.text_input("Notes (Optional)", placeholder="e.g. Harvest from Field B")
                    # -----------------------
                    
                    st.write(f"📈 New Balance: :green[{current_qty + add_qty:,.1f} g]")
                    
                    if st.form_submit_button("Confirm Addition", use_container_width=True):
                        # 1. Update Inventory Qty
                        update_batch_qty(selected_lot, add_qty)
                        
                        # 2. Log Transaction
                        current_user = st.session_state.get('name', 'Admin')
                        log_transaction(selected_lot, "IN", add_qty, reason_in, notes_in, current_user)
                        
                        st.success(f"✅ Added {add_qty}g & Logged!")
                        st.rerun()
            # === TAB 3: MOVE LOCATION (New!) ===
            with tab_move:
                st.caption("Update physical location in storage.")
                with st.form("move_stock_form"):
                    # Structured Inputs based on your Layout (3 Racks, 4 Rows, 6 Cols)
                    mc1, mc2, mc3 = st.columns(3)
                    
                    # Racks 1-3
                    new_rack = mc1.selectbox("Rack", ["Rack 1", "Rack 2", "Rack 3"])
                    
                    # Rows 1-4
                    new_row = mc2.selectbox("Row", [f"Row {i}" for i in range(1, 5)])
                    
                    # Columns 1-6
                    new_col = mc3.selectbox("Column", [f"Col {i}" for i in range(1, 7)])
                    
                    # Preview the formatted string
                    formatted_loc = f"{new_rack}, {new_row}, {new_col}"
                    st.info(f"New Location Tag: **{formatted_loc}**")
                    
                    if st.form_submit_button("Update Location", use_container_width=True):
                        # Logic to save ONLY the location
                        idx = df_inv.index[df_inv['lot_code'] == selected_lot][0]
                        df_inv.at[idx, 'location'] = formatted_loc
                        df_inv.at[idx, 'last_updated'] = datetime.now().strftime("%Y-%m-%d")
                        save_data(df_inv, "inventory")
                        
                        st.success(f"moved to {formatted_loc}")
                        st.rerun()


    # --- 3. RECEIVE STOCK (Mobile Friendly) ---
    elif page == "Add Seed Stock":
        st.title("📥 Receive New Stock")
        st.caption("Register new seed batches into the inventory.")

        # Load current data to check for duplicates
        df_inv = load_data("inventory")

        with st.form("receive_form", clear_on_submit=False):
            st.subheader("1. Batch Details")
            
            # Row 1: Identifiers
            c1, c2 = st.columns(2)
            lot_code = c1.text_input("Lot Code (Unique ID)", placeholder="e.g. L-2023-001")
            variety = c2.text_input("Variety Name", placeholder="e.g. Burley 21")
            
            # Row 2: Type & Year
            c3, c4 = st.columns(2)
            seed_type = c3.selectbox("Seed Type", ["Breeder Seed", "Foundation Seed", "Registered Seed", "Certified Seed"])
            year_prod = c4.number_input("Year Produced", min_value=2000, max_value=2030, value=2023)

            st.divider()
            st.subheader("2. Quantity & Quality")
            
            # Row 3: Weight & Germination
            c5, c6 = st.columns(2)
            qty_g = c5.number_input("Weight (grams)", min_value=0.0, value=1000.0, step=50.0)
            germ_rate = c6.number_input("Germination Rate (%)", min_value=0, max_value=100, value=95)

            st.divider()
            st.subheader("3. Storage Location")
            
            # Row 4: Structured Location (Matches your 3 racks x 4 rows x 6 cols layout)
            l1, l2, l3 = st.columns(3)
            rack = l1.selectbox("Rack", ["Rack 1", "Rack 2", "Rack 3"])
            row = l2.selectbox("Row", [f"Row {i}" for i in range(1, 5)])
            col = l3.selectbox("Column", [f"Col {i}" for i in range(1, 7)])
            
            # Combine into standard format string
            final_location = f"{rack}, {row}, {col}"
            st.info(f"📍 Storing at: **{final_location}**")

            st.divider()
            
            # SUBMIT BUTTON (Full Width for Mobile)
            submitted = st.form_submit_button("💾 Save to Inventory", type="primary", use_container_width=True)

            if submitted:
                # Validation 1: Check if Lot Code is empty
                if not lot_code or not variety:
                    st.error("❌ Error: Lot Code and Variety are required.")
                
                # Validation 2: Check for Duplicates
                elif not df_inv.empty and lot_code in df_inv['lot_code'].values:
                    st.error(f"❌ Error: Lot Code '{lot_code}' already exists! Please use a unique ID.")
                
                else:
                    # Create New Record
                    new_data = {
                        "lot_code": lot_code,
                        "variety": variety,
                        "type": seed_type,
                        "quantity_g": qty_g,
                        "current_germination": germ_rate,
                        "year_produced": year_prod,
                        "location": final_location,
                        "last_updated": datetime.now().strftime("%Y-%m-%d")
                    }
                    
                    # Append and Save
                    updated_df = pd.concat([df_inv, pd.DataFrame([new_data])], ignore_index=True)
                    save_data(updated_df, "inventory")
                    
                    # Log Transaction (Initial Stock In)
                    # We assume you have the log_transaction function from previous steps
                    current_user = st.session_state.get('name', 'Admin')
                    log_transaction(lot_code, "IN", qty_g, "Initial Stocking", "New batch received", current_user)
                    
                    st.success(f"✅ Batch {lot_code} saved successfully!")
                    
                    # Optional: Auto-generate label preview
                    st.balloons()


    # --- 4. ANALYTICS (With Usage Reports) ---
    elif page == "Analytics":
        st.title("📊 Inventory Intelligence")
        st.caption("Data-driven insights for stock, quality, and usage trends.")
        
        # Load All Data Sources
        df_inv = load_data("inventory")
        df_env_log = load_data("env_logs") 
        df_trans = load_data("transactions") # NEW: Load transaction history
        
        if df_inv.empty:
            st.warning("No inventory data found.")
        else:
            # --- 0. PRE-PROCESSING ---
            df_inv['quantity_g'] = pd.to_numeric(df_inv['quantity_g'], errors='coerce').fillna(0)
            df_inv['current_germination'] = pd.to_numeric(df_inv['current_germination'], errors='coerce').fillna(0)
            df_inv['year_produced'] = pd.to_numeric(df_inv['year_produced'], errors='coerce').fillna(0)
            
            # --- 1. KPI BOARD (Mobile Friendly) ---
            total_weight_kg = df_inv['quantity_g'].sum() / 1000.0
            avg_germ = df_inv['current_germination'].mean()
            
            # Mobile Grid Layout (2x2)
            k1, k2 = st.columns(2)
            k3, k4 = st.columns(2)
            
            with k1: st.metric("Total Stock", f"{total_weight_kg:.2f} kg")
            with k2: st.metric("Avg Germination", f"{avg_germ:.1f}%")
            with k3: 
                low_stock = len(df_inv[df_inv['quantity_g'] < 500])
                st.metric("Low Stock Batches", low_stock, delta_color="inverse")
            with k4: 
                crit = len(df_inv[df_inv['current_germination'] < 80])
                st.metric("Critical Quality", crit, delta_color="inverse")
            
            st.divider()

            # --- 2. ADVANCED TABS ---
            # Added "Transaction Log" as the 4th tab
            t1, t2, t3, t4 = st.tabs(["📦 Stock Info", "🌍 Environment", "⚠️ Actions", "📉 Transaction Log"])
            
            # === TAB 1: STOCK COMPOSITION ===
            with t1:
                st.subheader("Inventory Hierarchy")
                fig_sun = px.sunburst(
                    df_inv, 
                    path=['type', 'variety', 'lot_code'], 
                    values='quantity_g',
                    color='current_germination',
                    color_continuous_scale='RdYlGn',
                    range_color=[50, 100],
                    title="Stock Distribution (Color = Quality)"
                )
                st.plotly_chart(fig_sun, use_container_width=True)
                
            # === TAB 2: ENVIRONMENT ===
            with t2:
                st.subheader("Storage Conditions vs Quality")
                # (Existing logic for environment correlation)
                inv_stats = df_inv.groupby('year_produced')['current_germination'].mean().reset_index()
                
                if not df_env_log.empty:
                    df_env_log['date'] = pd.to_datetime(df_env_log['date'])
                    df_env_log['Year'] = df_env_log['date'].dt.year
                    env_stats = df_env_log.groupby('Year')[['avg_temp', 'avg_humidity']].mean().reset_index()
                    merged_df = pd.merge(inv_stats, env_stats, left_on='year_produced', right_on='Year', how='outer').sort_values('year_produced')
                else:
                    merged_df = inv_stats

                if not merged_df.empty:
                    from plotly.subplots import make_subplots
                    import plotly.graph_objects as go
                    fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_combo.add_trace(go.Bar(x=merged_df['year_produced'], y=merged_df['current_germination'], name="Germination %", marker_color='lightgreen'), secondary_y=False)
                    if 'avg_temp' in merged_df.columns:
                        fig_combo.add_trace(go.Scatter(x=merged_df['year_produced'], y=merged_df['avg_temp'], name="Avg Temp (°C)", mode='lines+markers', line=dict(color='red')), secondary_y=True)
                    st.plotly_chart(fig_combo, use_container_width=True)
                else:
                    st.info("Not enough data to correlate environment yet.")

            # === TAB 3: ACTION ITEMS ===
            with t3:
                c_disp, c_stale = st.columns(2)
                disposal = df_inv[df_inv['current_germination'] < 75]
                with c_disp:
                    st.error(f"🚨 Low Viability ({len(disposal)})")
                    if not disposal.empty: st.dataframe(disposal[['lot_code', 'variety', 'current_germination']], hide_index=True)
                
                # Simple stale check
                with c_stale:
                    st.warning("📅 Check Updates")
                    st.caption("Review these batches for re-testing.")
                    st.dataframe(df_inv.sort_values('last_updated').head(5)[['lot_code', 'last_updated']], hide_index=True)

            # === TAB 4: TRANSACTION LOG (NEW!) ===
            with t4:
                st.subheader("📜 Audit Trail & Usage Reports")
                
                if df_trans.empty:
                    st.info("No transaction history found yet. Start using 'Warehouse Mode' to generate logs.")
                else:
                    # 1. High-Level Metrics
                    # Filter for 'OUT' (Usage) vs 'IN' (Restock)
                    df_out = df_trans[df_trans['action'] == "OUT"]
                    df_in = df_trans[df_trans['action'] == "IN"]
                    
                    m_out = df_out['quantity_g'].sum()
                    m_in = df_in['quantity_g'].sum()
                    
                    tm1, tm2 = st.columns(2)
                    tm1.metric("Total Seeds Distributed/Used", f"{m_out:,.1f} g", delta="- Outflow", delta_color="inverse")
                    tm2.metric("Total Seeds Restocked", f"{m_in:,.1f} g", delta="+ Inflow")
                    
                    st.divider()

                    # 2. Visualizations
                    # A. Why are we removing seeds? (Reason Analysis)
                    if not df_out.empty:
                        st.markdown("##### 📉 Why are seeds leaving storage?")
                        c_chart1, c_chart2 = st.columns(2)
                        
                        with c_chart1:
                            # Pie Chart: Quantity by Reason
                            fig_reason = px.pie(
                                df_out, 
                                values='quantity_g', 
                                names='reason', 
                                title="Usage by Reason (Weight)",
                                hole=0.4,
                                color_discrete_sequence=px.colors.sequential.RdBu
                            )
                            st.plotly_chart(fig_reason, use_container_width=True)
                        
                        with c_chart2:
                            # Bar Chart: Which Variety is used most?
                            var_usage = df_out.groupby('variety')['quantity_g'].sum().reset_index().sort_values('quantity_g', ascending=True)
                            fig_var = px.bar(
                                var_usage, 
                                x='quantity_g', 
                                y='variety', 
                                orientation='h', 
                                title="Top Utilized Varieties",
                                text_auto='.2s'
                            )
                            st.plotly_chart(fig_var, use_container_width=True)
                    
                    # 3. Recent Log Table
                    st.divider()
                    st.markdown("##### 🕵️ Recent Activity Log")
                    
                    # Sort by latest first
                    df_display_trans = df_trans.sort_values('timestamp', ascending=False)
                    
                    # Mobile Friendly: Show only key columns
                    st.dataframe(
                        df_display_trans[['timestamp', 'action', 'quantity_g', 'variety', 'reason', 'user']],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "timestamp": "Time",
                            "quantity_g": st.column_config.NumberColumn("Qty (g)", format="%.1f"),
                            "action": st.column_config.Column("Type", width="small")
                        }
                    )


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