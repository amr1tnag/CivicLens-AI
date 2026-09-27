"""Central configuration: paths and the civic taxonomy (categories, departments, SLAs).

Keeping the taxonomy here (rather than scattered across modules) means the
category list, routing table, and SLA policy can be updated in one place
without touching model code.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

DATASET_PATH = RAW_DATA_DIR / "complaints_dataset.csv"
CATEGORY_MODEL_PATH = MODELS_DIR / "category_model.joblib"
URGENCY_MODEL_PATH = MODELS_DIR / "urgency_model.joblib"
DUPLICATE_INDEX_PATH = MODELS_DIR / "duplicate_index.joblib"

DB_PATH = BASE_DIR / "civiclens.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Complaint categories
# --------------------------------------------------------------------------
CATEGORIES = [
    "Roads & Infrastructure",
    "Water Supply",
    "Electricity & Power",
    "Sanitation & Waste Management",
    "Street Lighting",
    "Public Safety & Law Enforcement",
    "Parks & Public Property",
    "Noise & Pollution",
    "Health & Sanitation",
    "Other / General Administration",
]

# --------------------------------------------------------------------------
# Urgency levels, ordered from least to most urgent.
# `weight` is used to turn a predicted class back into a 0-100 priority score
# and to resolve ties (e.g. between model prediction and lexicon signal).
# `sla_hours` drives the routing engine's promised response time.
# --------------------------------------------------------------------------
URGENCY_LEVELS = ["Low", "Medium", "High", "Critical"]

URGENCY_SLA_HOURS = {
    "Critical": 4,
    "High": 24,
    "Medium": 72,
    "Low": 168,
}

URGENCY_WEIGHT = {
    "Low": 25,
    "Medium": 50,
    "High": 75,
    "Critical": 100,
}

# --------------------------------------------------------------------------
# Category -> responsible civic department (routing table)
# --------------------------------------------------------------------------
CATEGORY_TO_DEPARTMENT = {
    "Roads & Infrastructure": "Public Works Department (PWD)",
    "Water Supply": "Municipal Water Board",
    "Electricity & Power": "Electricity Distribution Company (DISCOM)",
    "Sanitation & Waste Management": "Sanitation & Solid Waste Management Dept.",
    "Street Lighting": "Electrical Maintenance Cell (PWD)",
    "Public Safety & Law Enforcement": "Local Police Department",
    "Parks & Public Property": "Parks & Horticulture Department",
    "Noise & Pollution": "Pollution Control Board",
    "Health & Sanitation": "Public Health Department",
    "Other / General Administration": "General Administration / Citizen Helpdesk",
}

# Departments that are always notified immediately regardless of category
# when a complaint is scored Critical (e.g. life-safety risk).
CRITICAL_ESCALATION_DEPARTMENT = "Disaster & Emergency Response Cell"

# Synthetic city zones used by the dataset generator and accepted by the API
# as a free-text `location` field describing where the issue was observed.
ZONES = [
    "Ward 1 - Riverside",
    "Ward 2 - Old Town",
    "Ward 3 - Lakeview",
    "Ward 4 - Hilltop",
    "Ward 5 - Market District",
    "Ward 6 - Greenfield",
    "Ward 7 - Industrial Area",
    "Ward 8 - North Colony",
    "Ward 9 - South Colony",
    "Ward 10 - Central Business District",
]
