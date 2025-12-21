# NTA Seed Management System

The **NTA Seed Management System** is a comprehensive, Streamlit-based application designed to manage and track seed storage activities for the National Tobacco Administration. It provides tools for inventory management, germination tracking, environmental monitoring, and seed stock distribution.

## 🚀 Features

*   **📊 Dashboard**: Real-time overview of total seed batches, varieties, stock weight, and low stock alerts. Includes an interactive **Virtual Warehouse Map** to visualize storage locations.
*   **📦 Inventory Management**:
    *   **Inventory List**: View, filter, and search seed batches (Table & Card views).
    *   **Batch Manager**: Edit batch details, print QR code labels (PDF), and manage stock.
*   **📱 Warehouse Mode**: A mobile-friendly interface optimized for warehouse staff to:
    *   Scan QR codes (via camera or upload).
    *   Quickly remove stock (usage/distribution).
    *   Add stock (returns/new harvest).
    *   Move batches to new storage locations.
*   **📥 Add Seed Stock**: Form-based entry for registering new seed batches with automatic duplicate checking.
*   **📈 Analytics**:
    *   **Stock Info**: Visualizations of stock distribution by type and variety.
    *   **Environment**: Correlation analysis between storage conditions (Temp/Humidity) and seed quality (Germination).
    *   **Transaction Log**: Audit trail of all stock movements (IN/OUT) and usage reasons.
*   **🌡️ Environment Monitor**:
    *   Upload and parse datalogger logs (CSV/XLSX).
    *   Analyze historical temperature and humidity trends.
    *   Sync daily summaries to the cloud for long-term tracking.
*   **🔐 Admin Panel**: Secure user management (Add/Remove users).
*   **🖨️ Label Printing**: Generate professional PDF labels (Thermal 4x2" or A4 Avery) with QR codes.

## 🛠️ Tech Stack

*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **Data Processing**: Pandas, NumPy
*   **Visualization**: Plotly Express
*   **Database**: Google Sheets (via `st-gsheets-connection`)
*   **Authentication**: `streamlit-authenticator`
*   **Utilities**: OpenCV (QR Scanning), ReportLab (PDF Generation)

## 📋 Prerequisites

*   Python 3.9+
*   A Google Cloud Platform project with the **Google Sheets API** and **Google Drive API** enabled.
*   A Google Service Account with access to the target Google Sheet and Google Drive folder.

## ⚙️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Secrets**:
    Create a file named `.streamlit/secrets.toml` in the project root. This file connects the app to your Google Cloud resources.

    ```toml
    [connections.gsheets]
    spreadsheet = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit"
    type = "service_account"
    project_id = "your-project-id"
    private_key_id = "your-private-key-id"
    private_key = "-----BEGIN PRIVATE KEY-----\n..."
    client_email = "your-service-account@..."
    client_id = "..."
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "..."

    [drive]
    folder_id = "YOUR_GOOGLE_DRIVE_FOLDER_ID"
    ```

4.  **Configure Users**:
    The `config.yaml` file stores user credentials. It is pre-configured with default users. You can manage users via the **Admin Panel** in the app.

## 🚀 Usage

Run the Streamlit application:

```bash
streamlit run svis_v2.py
```

The app will open in your default web browser.

## 📂 Project Structure

*   `svis_v2.py`: The main application entry point containing all UI and logic.
*   `requirements.txt`: Python package dependencies.
*   `config.yaml`: User authentication configuration.
*   `init_db_v2.py`: (Optional) Script for initializing a local SQLite database (legacy/backup use).
*   `auth_utils.py`: Helper functions for authentication.
*   `logo.png`: Application logo.

## 🛡️ Authentication

The system uses `streamlit-authenticator` for secure login.
*   **Default Admin**: `admin` / `admin123` (Check `config.yaml` for actual hash).
*   Passwords are hashed using bcrypt.

## 📝 Notes

*   **Google Sheets Structure**: The app expects specific worksheets in the connected Google Sheet: `inventory`, `history`, `transactions`, `env_logs`, `raw_logs`.
*   **Label Printing**: Designed for 4x2 inch thermal labels or Standard A4 Avery sheets.
