# GreenLeaf Manager / GoldenLeafSeeds

A comprehensive Seed Inventory Management System designed to track seed batches, manage germination tests, and visualize inventory data. This application is built with Python and Streamlit, providing a user-friendly interface for warehouse managers.

## Features

- **Authentication**: Secure admin login to access the dashboard.
- **Dashboard**:
    - **Inventory List**: View all seed batches with real-time status indicators (e.g., Critical, Low Stock).
    - **Filtering & Search**: Quickly find batches by variety or lot code.
- **Batch Management**:
    - **Inspect & History**: View detailed batch information and historical germination test results.
    - **Edit Details**: Update batch metadata such as quantity, location, or variety.
    - **Add Test Result**: Record new germination tests, which automatically updates the batch's current status and history log.
    - **Delete**: Permanently remove batches and their history.
- **Receive New Batch**: specialized form for entering new seed inventory into the system.
- **Analytics**: Visualizations for inventory distribution and viability trends.

## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Database**: SQLite
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/)
- **Visualization**: [Plotly](https://plotly.com/)
- **Authentication**: Custom hash-based verification

## Prerequisites

- Python 3.8 or higher

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Database Initialization

Before running the application, you need to initialize the database. This process imports the initial data from `seed-inventory.csv`, handles deduplication, and sets up the history tracking tables.

Run the initialization script:
```bash
python init_db_v2.py
```
This will create (or reset) the `inventory_v2.db` SQLite database file.

## Running the Application

To start the GreenLeaf Manager dashboard, run:

```bash
streamlit run svis_v2.py
```

The application will open in your default web browser (usually at `http://localhost:8501`).

## Default Credentials

To log in to the admin dashboard, use the following credentials:

- **Username**: `admin`
- **Password**: `admin123`

## Project Structure

- `svis_v2.py`: The main Streamlit application file containing the UI and business logic.
- `init_db_v2.py`: Script to initialize the SQLite database and migrate data from the CSV.
- `auth_utils.py`: Helper functions for password hashing and verification.
- `seed-inventory.csv`: Initial dataset used for seeding the database.
- `requirements.txt`: Python dependencies.
