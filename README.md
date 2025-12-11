# GoldenLeafSeeds

## GreenLeaf Manager

A seed inventory management system built with Streamlit, Pandas, and SQLite.

### Setup

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the application:
    ```bash
    streamlit run svis_v2.py
    ```

    The application will automatically initialize the database (`inventory_v2.db`) from `seed-inventory.csv` if it does not exist.

### Authentication

The system is protected by a login page.
*   **Username**: `admin`
*   **Password**: `admin123`

### Features

*   **Dashboard**: View inventory, filter by variety/lot, and manage batches.
*   **Batch Manager**: Edit details, record germination tests, and delete batches.
*   **Receive New Batch**: Add new seed lots to the system.
*   **Analytics**: Visualize inventory distribution and viability.
