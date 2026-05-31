"""
utils/feature_engineering.py — V2
─────────────────────────────────────────────────────────────────────
Géron-Style Feature Engineering Pipeline for Startup Classification.

V2 Changes:
  - Added 7 new features: funding_rounds_sq, funding_duration_days,
    funding_recency_years, funding_velocity_log (replaces raw velocity),
    rounds_per_year, is_top_ipo_region, is_biotech, has_founded_date
  - Log-transformed funding_per_round (funding_per_round_log)
  - Log-transformed funding_total (funding_log) instead of raw
  - Total: 21 features (11 numeric + 7 binary + 3 categorical)

Architecture:
─────────────
  Raw CSV (14 columns)
    │
    ├─► RawCleaner          → fix types, parse dates, filter target
    ├─► FeatureEngineerV2   → create derived features (row-level)
    ├─► CategoryGrouper     → group rare categories into "Other"
    │
    └─► ColumnTransformer (fit on train only)
          ├── numeric_pipe   → Imputer → RobustScaler
          ├── categorical_pipe → Imputer → OrdinalEncoder
          └── passthrough    → binary flags (already 0/1)

Authors: Mohamed Belfilali Mimoun / Walid Fadli
Project: PFA 2025–2026 · ENSAM-Rabat
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS — V2
# ═══════════════════════════════════════════════════════════════════

NUM_COLS = [
    "funding_log",                 # log1p(funding_total_usd)
    "funding_rounds",              # raw count
    "funding_per_round_log",       # log1p(funding_total / rounds)
    "funding_rounds_sq",           # funding_rounds²
    "company_age_years",           # last_funding_year - founded_year
    "time_to_first_funding_days",  # first_funding - founded (days)
    "funding_duration_days",       # last_funding - first_funding (days)
    "funding_recency_years",       # REFERENCE_YEAR - last_funding_year
    "funding_velocity_log",        # log1p(funding / age)
    "rounds_per_year",             # funding_rounds / age
    "category_count",              # number of | in category_list
]

CAT_COLS = [
    "primary_category_grouped",
    "country_grouped",
    "region_grouped",
]

BINARY_COLS = [
    "has_funding",                 # funding > 0
    "is_us_startup",               # country == USA
    "is_top_ipo_region",           # region in top IPO regions
    "is_biotech",                  # biotech/pharma category
    "has_multiple_categories",     # category_count > 1
    "has_website",                 # homepage_url not null
    "has_founded_date",            # founded_at not null
]

# No PASSTHROUGH_COLS — everything is either numeric, binary, or categorical
ALL_FEATURE_COLS = NUM_COLS + CAT_COLS + BINARY_COLS

# Feature groups for ablation studies
FEATURE_GROUPS = {
    "funding": ["funding_log", "funding_per_round_log", "funding_rounds",
                "funding_rounds_sq", "has_funding"],
    "timing": ["company_age_years", "time_to_first_funding_days",
               "funding_duration_days", "funding_recency_years", "has_founded_date"],
    "velocity": ["funding_velocity_log", "rounds_per_year"],
    "geographic": ["is_us_startup", "is_top_ipo_region", "country_grouped", "region_grouped"],
    "category": ["primary_category_grouped", "category_count",
                 "has_multiple_categories", "is_biotech"],
    "other": ["has_website"],
}

# Defaults
DEFAULT_TOP_N_COUNTRIES = 10
DEFAULT_TOP_N_CATEGORIES = 20
DEFAULT_TOP_N_REGIONS = 15

REFERENCE_YEAR = 2015  # dataset era midpoint (most data is 2005-2015)
VALID_STATUSES = {"operating", "closed", "acquired", "ipo"}


# ═══════════════════════════════════════════════════════════════════
# TRANSFORMER 1 — RawCleaner (unchanged from V1)
# ═══════════════════════════════════════════════════════════════════

class RawCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans the raw DataFrame: fix types, parse dates, normalize text.
    No fit() state — all operations are deterministic and row-level.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()

        if "funding_total_usd" in df.columns:
            df["funding_total_usd"] = (
                df["funding_total_usd"]
                .astype(str).str.strip()
                .replace("-", np.nan)
                .replace("nan", np.nan)
            )
            df["funding_total_usd"] = pd.to_numeric(
                df["funding_total_usd"], errors="coerce"
            )

        for col in ["founded_at", "first_funding_at", "last_funding_at"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                mask = df[col].dt.year.between(1900, 2025)
                df.loc[~mask, col] = pd.NaT

        if "status" in df.columns:
            df["status"] = df["status"].str.strip().str.lower()

        if "name" in df.columns:
            df = df.dropna(subset=["name"])

        return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# TRANSFORMER 2 — FeatureEngineerV2
# ═══════════════════════════════════════════════════════════════════

class FeatureEngineerV2(BaseEstimator, TransformerMixin):
    """
    Creates all V2 derived features from the cleaned DataFrame.

    All transformations are ROW-LEVEL — no cross-row statistics.
    Safe to call before splitting.

    Produces 21 feature columns organized in 6 groups:
      - Funding (5): funding_log, funding_rounds, funding_per_round_log,
                     funding_rounds_sq, has_funding
      - Timing (5): company_age_years, time_to_first_funding_days,
                    funding_duration_days, funding_recency_years, has_founded_date
      - Velocity (2): funding_velocity_log, rounds_per_year
      - Geographic (4): is_us_startup, is_top_ipo_region, country_grouped*, region_grouped*
      - Category (4): primary_category_grouped*, category_count,
                     has_multiple_categories, is_biotech
      - Other (1): has_website

    (*) categorical columns created by CategoryGrouper, not here
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()

        # ── Group 1: Funding features ──
        funding = df["funding_total_usd"].fillna(0) if "funding_total_usd" in df.columns else 0
        df["funding_log"] = np.log1p(funding)

        rounds = df["funding_rounds"].fillna(0) if "funding_rounds" in df.columns else 0
        fpr = np.where(rounds > 0, funding / rounds, 0)
        df["funding_per_round_log"] = np.log1p(fpr)

        df["has_funding"] = (funding > 0).astype(int)
        df["funding_rounds_sq"] = rounds ** 2

        # ── Group 2: Timing features ──
        if "founded_at" in df.columns:
            df["founded_year"] = df["founded_at"].dt.year
        else:
            df["founded_year"] = np.nan

        last_year = (
            df["last_funding_at"].dt.year.fillna(REFERENCE_YEAR)
            if "last_funding_at" in df.columns else REFERENCE_YEAR
        )
        founded_year = df["founded_year"].fillna(REFERENCE_YEAR)

        df["company_age_years"] = (last_year - founded_year).clip(lower=0)

        if "first_funding_at" in df.columns and "founded_at" in df.columns:
            df["time_to_first_funding_days"] = (
                (df["first_funding_at"] - df["founded_at"]).dt.days.clip(lower=0)
            )
        else:
            df["time_to_first_funding_days"] = np.nan

        if "last_funding_at" in df.columns and "first_funding_at" in df.columns:
            df["funding_duration_days"] = (
                (df["last_funding_at"] - df["first_funding_at"]).dt.days.clip(lower=0)
            )
        else:
            df["funding_duration_days"] = np.nan

        if "last_funding_at" in df.columns:
            df["funding_recency_years"] = (
                REFERENCE_YEAR - df["last_funding_at"].dt.year.fillna(REFERENCE_YEAR)
            ).clip(lower=0)
        else:
            df["funding_recency_years"] = 0

        df["has_founded_date"] = df["founded_at"].notna().astype(int) if "founded_at" in df.columns else 0

        # ── Group 3: Velocity features ──
        age_safe = df["company_age_years"].clip(lower=1)
        df["funding_velocity_log"] = np.log1p(funding / age_safe)
        df["rounds_per_year"] = rounds / age_safe

        # ── Group 4: Geographic features (binary part — categorical part is in CategoryGrouper) ──
        if "country_code" in df.columns:
            df["is_us_startup"] = (
                df["country_code"].str.strip().str.upper() == "USA"
            ).astype(int)
        else:
            df["is_us_startup"] = 0

        top_ipo_regions = ["SF Bay", "New York", "Boston", "London"]
        if "region" in df.columns:
            df["is_top_ipo_region"] = df["region"].fillna("").isin(top_ipo_regions).astype(int)
        else:
            df["is_top_ipo_region"] = 0

        # ── Group 5: Category features (binary/numeric part) ──
        if "category_list" in df.columns:
            df["primary_category"] = (
                df["category_list"].fillna("Unknown").str.split("|").str[0].str.strip()
            )
            df["category_count"] = (
                df["category_list"].fillna("")
                .apply(lambda x: len(x.split("|")) if x else 0)
            )
            df["has_multiple_categories"] = (df["category_count"] > 1).astype(int)

            biotech_kw = ["biotech", "biotechnology", "health care", "pharmaceuticals"]
            df["is_biotech"] = df["category_list"].fillna("").str.lower().apply(
                lambda x: int(any(k in x for k in biotech_kw))
            )
        else:
            df["primary_category"] = "Unknown"
            df["category_count"] = 1
            df["has_multiple_categories"] = 0
            df["is_biotech"] = 0

        # ── Group 6: Other ──
        df["has_website"] = df["homepage_url"].notna().astype(int) if "homepage_url" in df.columns else 0

        return df


# ═══════════════════════════════════════════════════════════════════
# TRANSFORMER 3 — CategoryGrouper (unchanged from V1)
# ═══════════════════════════════════════════════════════════════════

class CategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Groups rare categories into "Other" for high-cardinality columns.
    HAS state: learns top-N values during fit().
    """

    def __init__(
        self,
        top_n_countries=DEFAULT_TOP_N_COUNTRIES,
        top_n_categories=DEFAULT_TOP_N_CATEGORIES,
        top_n_regions=DEFAULT_TOP_N_REGIONS,
    ):
        self.top_n_countries = top_n_countries
        self.top_n_categories = top_n_categories
        self.top_n_regions = top_n_regions

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        if "country_code" in df.columns:
            self.top_countries_ = (
                df["country_code"].str.strip().str.upper()
                .value_counts().nlargest(self.top_n_countries).index.tolist()
            )
        else:
            self.top_countries_ = []

        if "primary_category" in df.columns:
            self.top_categories_ = (
                df["primary_category"].value_counts()
                .nlargest(self.top_n_categories).index.tolist()
            )
        else:
            self.top_categories_ = []

        if "region" in df.columns:
            self.top_regions_ = (
                df["region"].dropna().str.strip()
                .value_counts().nlargest(self.top_n_regions).index.tolist()
            )
        else:
            self.top_regions_ = []

        return self

    def transform(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        if "country_code" in df.columns:
            cc = df["country_code"].str.strip().str.upper()
            df["country_grouped"] = cc.where(cc.isin(self.top_countries_), "Other")
        else:
            df["country_grouped"] = "Other"

        if "primary_category" in df.columns:
            df["primary_category_grouped"] = df["primary_category"].where(
                df["primary_category"].isin(self.top_categories_), "Other"
            )
        else:
            df["primary_category_grouped"] = "Other"

        if "region" in df.columns:
            rg = df["region"].fillna("Unknown").str.strip()
            df["region_grouped"] = rg.where(rg.isin(self.top_regions_), "Other")
        else:
            df["region_grouped"] = "Other"

        return df


# ═══════════════════════════════════════════════════════════════════
# TRANSFORMER 4 — FeatureSelector (unchanged)
# ═══════════════════════════════════════════════════════════════════

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Selects only the columns needed by the ColumnTransformer."""

    def __init__(self, columns=None):
        self.columns = columns or ALL_FEATURE_COLS

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col in self.columns:
            if col not in df.columns:
                df[col] = np.nan
        return df[self.columns].copy()


# ═══════════════════════════════════════════════════════════════════
# PIPELINE BUILDER — V2
# ═══════════════════════════════════════════════════════════════════

def build_preprocessing_pipeline(**kwargs):
    """
    Build the ColumnTransformer for V2 features.

    V2 differences from V1:
      - No log1p inside the pipeline (already done in FeatureEngineerV2)
      - Binary features go through passthrough (no imputation needed, already 0/1)
      - 11 numeric + 3 categorical + 7 binary = 21 features
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler, OrdinalEncoder

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUM_COLS),
            ("cat", categorical_pipe, CAT_COLS),
            ("bin", "passthrough", BINARY_COLS),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_names():
    """Return ordered feature names after preprocessing."""
    return (
        [f"num_{c}" for c in NUM_COLS]
        + [f"cat_{c}" for c in CAT_COLS]
        + BINARY_COLS
    )
