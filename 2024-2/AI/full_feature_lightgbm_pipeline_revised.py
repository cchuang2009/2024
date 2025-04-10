


# --- Utility: Signed log1p transform ---
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))

# --- Raw Feature Engineering First ---
def add_raw_features(df):
    df = TI_new(df)
    df = derive_broker_features(df, n=3)
    df = derive_institutional_financial_features(df)
    df = derive_monthly_revenue_features(df, quantile_labeled=False)
    df = derive_chips_features(df)
    return df

# --- Then Apply Encoding (log/qcut) ---
def enhance_features(df, columns_to_transform):
    for col in columns_to_transform:
        if col not in df.columns:
            continue
        try:
            df[col + '_signedlog'] = signed_log1p(df[col])
            df[col + '_qcut'] = pd.qcut(df[col], q=100, labels=False, duplicates='drop')
        except Exception:
            df[col + '_qcut'] = np.nan
    return df

# --- Unified Feature Pipeline ---
def add_all_features(df, quantile_labeled=True, transform_columns=None):
    df = add_raw_features(df)
    if transform_columns:
        df = enhance_features(df, transform_columns)
    return df
