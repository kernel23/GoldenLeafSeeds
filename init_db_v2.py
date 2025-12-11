import pandas as pd
import sqlite3
import os
import random
from datetime import datetime, timedelta

DB_FILE = "inventory_v2.db"

def init_db():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE) # Reset for this demo

    # 1. Load your CSV
    try:
        df = pd.read_csv('seed-inventory.csv')
    except FileNotFoundError:
        print("❌ 'seed-inventory.csv' not found.")
        return

    # Clean / Rename columns
    df.rename(columns={
        'Type': 'type', 'Variety': 'variety', 'Lot Code': 'lot_code',
        'Quantity on Hand (g)': 'quantity_g', 'Year Produced': 'year_produced',
        'Germination Rate (%)': 'germination_rate', 'Storage Location': 'location',
        'Last Updated': 'last_updated'
    }, inplace=True)

    # --- DEDUPLICATION LOGIC ---
    print(f"Original rows: {len(df)}")
    # 1. Deduplicate exact duplicates (ignoring Last Updated if everything else matches)
    cols_to_check = ['lot_code', 'type', 'variety', 'quantity_g', 'year_produced', 'germination_rate', 'location']
    df = df.drop_duplicates(subset=cols_to_check)
    print(f"After dropping exact duplicates: {len(df)}")

    # 2. Handle remaining duplicates in lot_code (Split lots)
    # Find duplicates
    dupes = df[df.duplicated('lot_code', keep=False)].sort_values('lot_code')
    if not dupes.empty:
        print("Handling split lots (appending suffix to duplicates):")
        # Group by lot_code
        for lot, group in dupes.groupby('lot_code'):
            # Skip the first one
            for i, (idx, row) in enumerate(group.iloc[1:].iterrows(), start=2):
                 new_lot = f"{lot}-{i}"
                 df.at[idx, 'lot_code'] = new_lot
                 print(f"  Renamed {lot} to {new_lot}")
    # ---------------------------

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 2. Create Tables
    # TABLE 1: Main Inventory (The current state)
    cursor.execute("""
    CREATE TABLE inventory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lot_code TEXT UNIQUE,
        type TEXT,
        variety TEXT,
        quantity_g INTEGER,
        year_produced INTEGER,
        current_germination INTEGER,
        location TEXT,
        last_updated TEXT
    )
    """)

    # TABLE 2: Germination History (The log of tests)
    cursor.execute("""
    CREATE TABLE germination_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        inventory_id INTEGER,
        test_date TEXT,
        rate INTEGER,
        notes TEXT,
        FOREIGN KEY(inventory_id) REFERENCES inventory(id)
    )
    """)

    # 3. Populate Data & Simulate History
    print("⚙️  Migrating data and simulating history...")

    for _, row in df.iterrows():
        # Insert into Inventory
        try:
            cursor.execute("""
                INSERT INTO inventory (lot_code, type, variety, quantity_g, year_produced, current_germination, location, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (row['lot_code'], row['type'], row['variety'], row['quantity_g'], row['year_produced'], row['germination_rate'], row['location'], row['last_updated']))

            inventory_id = cursor.lastrowid

            # --- SIMULATE HISTORY ---
            # Event A: Harvest (Oct 1st of production year) - Assume 98% Viability
            harvest_date = f"{row['year_produced']}-10-01"
            cursor.execute("INSERT INTO germination_logs (inventory_id, test_date, rate, notes) VALUES (?, ?, ?, ?)",
                        (inventory_id, harvest_date, 98, "Initial Harvest Test"))

            # Event B: The current status from your CSV
            # We try to parse the date, or default to today if messy
            try:
                current_date_obj = pd.to_datetime(row['last_updated'], format='%m/%d/%y, %I:%M %p')
                current_date = current_date_obj.strftime('%Y-%m-%d')
            except:
                current_date = datetime.now().strftime('%Y-%m-%d')

            cursor.execute("INSERT INTO germination_logs (inventory_id, test_date, rate, notes) VALUES (?, ?, ?, ?)",
                        (inventory_id, current_date, row['germination_rate'], "Most Recent Lab Test"))
        except sqlite3.IntegrityError as e:
            print(f"⚠️ Error inserting {row['lot_code']}: {e}")

    conn.commit()
    conn.close()
    print(f"✅ Database upgraded! Created '{DB_FILE}' with history tracking.")

if __name__ == "__main__":
    init_db()
