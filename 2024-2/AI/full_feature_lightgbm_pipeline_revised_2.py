
import pandas as pd
import numpy as np
from scipy.stats import rankdata, norm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

# --- XiCor 計算 ---
def xicor_vec(x, y, ties="auto"):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = len(y)
    if len(x) != n:
        raise IndexError(f"x, y length mismatch: {len(x)}, {len(y)}")
    if ties == "auto":
        ties = len(np.unique(y)) < n
    elif not isinstance(ties, bool):
        raise ValueError(f"expected ties either 'auto' or boolean, got {ties} instead")
    y = y[np.argsort(x)]
    r = rankdata(y, method="ordinal")
    nominator = np.sum(np.abs(np.diff(r)))
    if ties:
        l = rankdata(y, method="max")
        denominator = 2 * np.sum(l * (n - l))
        nominator *= n
    else:
        denominator = n**2 - 1
        nominator *= 3
    statistic = 1 - nominator / denominator
    p_value = norm.sf(statistic, scale=2 / 5 / np.sqrt(n))
    return statistic, p_value

# --- 建議特徵生成 ---
def create_suggested_features(df):
    new_features = {}
    if '技術指標_週MACD' in df.columns and '技術指標_週DIF' in df.columns:
        new_features['技術指標_黃金交叉'] = (df['技術指標_週MACD'] > df['技術指標_週DIF']).astype(int)
    for key in ['技術指標_週RSI(5)', '技術指標_週RSI(10)', '技術指標_週MACD']:
        if key not in df.columns:
            break
    else:
        new_features['技術指標_三線共振'] = (
            (df['技術指標_週RSI(5)'] > 50) &
            (df['技術指標_週RSI(10)'] > 50) &
            (df['技術指標_週MACD'] > 0)
        ).astype(int)
    if '外資券商_分點吃貨比(%)' in df.columns and '外資券商_分點出貨比(%)' in df.columns:
        new_features['外資券商_淨吃貨比'] = df['外資券商_分點吃貨比(%)'] - df['外資券商_分點出貨比(%)']
    if '日外資_外資持股比率(%)' in df.columns and '日外資_外資尚可投資比率(%)' in df.columns:
        new_features['外資_持股與上限差'] = df['日外資_外資尚可投資比率(%)'] - df['日外資_外資持股比率(%)']
    if '月營收_單月合併營收年成長(%)' in df.columns and '月營收_單月合併營收月變動(%)' in df.columns:
        new_features['月營收_成長交叉'] = (
            (df['月營收_單月合併營收年成長(%)'] > 0) &
            (df['月營收_單月合併營收月變動(%)'] > 0)
        ).astype(int)
    if '個股主力買賣超統計_近1日主力買賣超(%)' in df.columns and '個股主力買賣超統計_近5日主力買賣超(%)' in df.columns:
        new_features['主力買超趨勢上升'] = (
            df['個股主力買賣超統計_近1日主力買賣超(%)'] > df['個股主力買賣超統計_近5日主力買賣超(%)']
        ).astype(int)
    for key, series in new_features.items():
        df[key] = series
    return df, list(new_features.keys())

# --- 全流程特徵處理 + 分箱 + xicor ---
def full_feature_analysis_pipeline(chunk, target_col, bins=10):
    chunk['nan_count'] = chunk.isna().sum(axis=1)
    chunk, suggested_feature_names = create_suggested_features(chunk)
    features = chunk.drop(columns=[target_col], errors='ignore')
    target = chunk[target_col]
    xicor_results = []
    for col in features.columns:
        try:
            filled = features[col].fillna(features[col].median())
            binned = pd.qcut(filled, q=bins, duplicates='drop', labels=False)
            score, pval = xicor_vec(binned, target)
            xicor_results.append({
                'feature': col,
                'xicor': score,
                'p_value': pval,
                'is_suggested': col in suggested_feature_names
            })
        except Exception:
            continue
    return pd.DataFrame(xicor_results)

# --- 特徵篩選 for ML 模型 ---
def select_top_n_features(xicor_df, top_n=100, xicor_threshold=None, exclude_nan_count=True):
    df = xicor_df.copy()
    if exclude_nan_count:
        df = df[df["feature"] != "nan_count"]
    if xicor_threshold is not None:
        df = df[df["xicor"] >= xicor_threshold]
    df_sorted = df.sort_values("xicor", ascending=False)
    top_features = df_sorted.head(top_n)["feature"].tolist()
    return top_features

# --- LightGBM 模型訓練 ---
import os
import pickle  # Import pickle for model serialization
from datetime import datetime  # Import datetime to get current date

def train_lightgbm_cv(df, features, target_col='飆股', n_splits=5):
    X = df[features]
    y = df[target_col]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    aucs = []
    f1_scores = []
    
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)  # Create 'models' folder if not present

    # Get current date string in YYYYMMDD format for filenames
    date_str = datetime.now().strftime("%Y%m%d")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params={
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1
            },
            train_set=dtrain,
            valid_sets=[dtrain, dval],
            num_boost_round=1000
            # early_stopping_rounds=50,
            # verbose_eval=False
        )

        val_preds = model.predict(X_val)
        auc = roc_auc_score(y_val, val_preds)
        aucs.append(auc)
        # f1_ = f1_score(y_val, (val_preds > 0.5).astype(int))
        # f1_scores.append(f1_)

        # Save the trained model for this fold to disk
        filename = f"model-{date_str}-fold-{fold+1}.pkl"
        filepath = os.path.join("models", filename)
        with open(filepath, "wb") as f:
            pickle.dump(model, f)

        models.append(model)
        print(f"Fold {fold+1} AUC: {auc:.4f}")
        # print(f"Fold {fold+1} F1: {f1_:.4f}")

    print(f"Mean AUC: {np.mean(aucs):.4f}")
    # print(f"Mean F1: {np.mean(f1_scores):.4f}")
    return models

import matplotlib.pyplot as plt

# --- 自動特徵選擇 + 報告 + 視覺化 ---
def auto_select_features_report(xicor_df, xicor_threshold=0.5, exclude_nan_count=True, plot=True):
    """
    根據 XiCor 門檻，自動選特徵，並回報統計與圖表

    回傳：
    - selected_features: List[str]
    """
    df = xicor_df.copy()
    if exclude_nan_count:
        df = df[df["feature"] != "nan_count"]
    total = len(df)
    above_thresh = df[df["xicor"] >= xicor_threshold]
    selected_features = above_thresh["feature"].tolist()
    count_selected = len(selected_features)

    print(f"Total features evaluated: {total}")
    print(f"Features with xicor ≥ {xicor_threshold}: {count_selected}")

    if plot:
        plt.figure(figsize=(10, 5))
        plt.hist(df["xicor"], bins=50, color='skyblue', edgecolor='k')
        plt.axvline(xicor_threshold, color='red', linestyle='--', label=f'Threshold = {xicor_threshold}')
        plt.title("XiCor 分布與選擇門檻")
        plt.xlabel("XiCor 值")
        plt.ylabel("特徵數量")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return selected_features

import joblib
import os

# --- 模型儲存 ---
def save_lightgbm_models(models, output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    for i, model in enumerate(models):
        joblib.dump(model, os.path.join(output_dir, f"lgbm_model_fold{i+1}.pkl"))
    print(f"Models saved to: {output_dir}")

# --- 預測與合併 ---
def predict_with_cv_models(models, X):
    preds = np.mean([model.predict(X) for model in models], axis=0)
    return preds

# --- 特徵重要性圖表 ---
def plot_feature_importance(models, feature_names, top_n=30):
    importances = np.mean([model.feature_importance() for model in models], axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color="teal")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import joblib
import os
from datetime import datetime
import pickle


# 分批訓練模型
def train_split_models(data, target_col='飆股', n_splits=20, top_feats=None):
    split_models = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for i, (train_idx, _) in enumerate(skf.split(data, data[target_col])):
        df_split = data.iloc[train_idx].copy()
        print(f"Training split {i+1}/{n_splits}, samples: {len(df_split)}")

        # 特徵處理 + XiCor
        xicor_df = full_feature_analysis_pipeline(df_split, target_col=target_col)
        feats = top_feats or select_top_n_features(xicor_df, top_n=100)
        df_split = df_split.dropna(subset=feats + [target_col])

        # 訓練單一 LightGBM 模型
        X = df_split[feats]
        y = df_split[target_col]
        dtrain = lgb.Dataset(X, label=y)

        model = lgb.train(
            params={'objective': 'binary', 'metric': 'auc', 'verbosity': -1},
            train_set=dtrain,
            num_boost_round=300
        )
        split_models.append((model, feats))

    return split_models

# 集成預測（平均）
def ensemble_predict(models_feats, df):
    preds = []
    for model, feats in models_feats:
        X = df[feats].copy()
        X = X.fillna(X.median(numeric_only=True))
        preds.append(model.predict(X))
    return np.mean(preds, axis=0)

# stacking 組合
def stacking_with_lr(models_feats, df, target_col='飆股'):
    meta_X = []
    y = df[target_col]
    
    for model, feats in models_feats:
        X = df[feats].copy()
        X = X.fillna(X.median(numeric_only=True))
        pred = model.predict(X)
        meta_X.append(pred)

    meta_X = np.array(meta_X).T
    meta_model = LogisticRegression()
    meta_model.fit(meta_X, y)
    print("Stacking model trained.")
    return meta_model

# stacking 預測
def stacking_predict(meta_model, models_feats, df):
    meta_X = []
    for model, feats in models_feats:
        X = df[feats].copy()
        X = X.fillna(X.median(numeric_only=True))
        pred = model.predict(X)
        meta_X.append(pred)
    meta_X = np.array(meta_X).T
    return meta_model.predict_proba(meta_X)[:, 1]

def TI_new(df):
    """
    衍生技術指標特徵，適用於已經經過 qcut 標籤處理（0~99）之資料。
    包含黃金交叉、乖離率趨勢、三線共振等。
    """
    # 黃金交叉：MACD > DIF
    if '技術指標_週MACD' in df.columns and '技術指標_週DIF' in df.columns:
        df['週黃金交叉'] = (df['技術指標_週MACD'] > df['技術指標_週DIF']).astype(int)

    # 乖離率變化趨勢（支援多個週期）
    for period in ['20日', '60日', '250日']:
        col = f'技術指標_乖離率({period})'
        if col in df.columns:
            df[f'{col}_上升'] = df[col].diff().gt(0).astype(int)

    # 多頭與空頭共振（RSI(5)、RSI(10)、MACD 都同向）
    rsi5, rsi10, macd = '技術指標_週RSI(5)', '技術指標_週RSI(10)', '技術指標_週MACD'
    if all(col in df.columns for col in [rsi5, rsi10, macd]):
        df['三線多頭共振'] = (
            (df[rsi5] > 50) & (df[rsi10] > 50) & (df[macd] > 50)
        ).astype(int)
        df['三線空頭共振'] = (
            (df[rsi5] < 50) & (df[rsi10] < 50) & (df[macd] < 50)
        ).astype(int)

    return df

#df = TI_new(df)
#df_test = TI_new(df_test)

def derive_broker_features(df, n=3):
    """
    衍生外資/主力券商分點相關特徵。
    包含買賣變化率、吃貨差值、連續買超天數（滯後特徵）。
    """
    # 1. 買賣總額變化率（使用前 n 天數據）
    for i in range(1, n + 1):
        for col in ['外資券商_分點進出', '外資券商_分點買賣力']:
            past_col = f'外資券商_前{i}天{col.split("_")[-1]}'
            if past_col in df.columns and col in df.columns:
                df[f'{col}_變化率_{i}日'] = (df[col] - df[past_col]) / (abs(df[past_col]) + 1e-6)

    # 2. 吃貨比 - 出貨比（淨吃貨比）
    if '外資券商_分點吃貨比(%)' in df.columns and '外資券商_分點出貨比(%)' in df.columns:
        df['外資券商_淨吃貨比'] = df['外資券商_分點吃貨比(%)'] - df['外資券商_分點出貨比(%)']

    # 3. 連續買超天數（以正買超記錄連續天數）
    buy_cols = [f'外資券商_前{i}天分點進出' for i in range(1, n + 1) if f'外資券商_前{i}天分點進出' in df.columns]
    if buy_cols:
        #buy_flags = (df[buy_cols] > 0).astype(int)
        buy_flags = (df[buy_cols] > 50).astype(int)
        df['外資券商_連續買超天數'] = buy_flags.sum(axis=1)

    return df
#df = derive_broker_features(df, n=3)
#df_test = derive_broker_features(df_test, n=3)

def derive_institutional_financial_features(df):
    """
    衍生法人持股與財務相關的高階特徵：
    - 投資額度差距
    - 毛利率 QoQ / YoY 變動率
    - 淨利與營收成長差（營利差）
    - 資產報酬率 x 負債比 交互作用
    """
    # 1. 外資可投資差距
    if '日外資_外資尚可投資比率(%)' in df.columns and '日外資_外資持股比率(%)' in df.columns:
        df['外資_持股與上限差'] = df['日外資_外資尚可投資比率(%)'] - df['日外資_外資持股比率(%)']

    # 2. 毛利率變動率 QoQ / YoY
    if '季IFRS財報_毛利率(%)' in df.columns and '季IFRS財報_毛利率累季(%)' in df.columns:
        df['毛利率_QoQ變動'] = df['季IFRS財報_毛利率(%)'].pct_change().fillna(0)
        df['毛利率_YoY變動'] = df['季IFRS財報_毛利率累季(%)'].pct_change(4).fillna(0)

    # 3. 營利差：淨利成長率 - 營收成長率
    net_profit_growth = '季IFRS財報_稅後純益成長率(%)'
    revenue_growth = '季IFRS財報_營收成長率(%)'
    if net_profit_growth in df.columns and revenue_growth in df.columns:
        df['營利差'] = df[net_profit_growth] - df[revenue_growth]

    # 4. 資產報酬率 x 負債比
    roaa = '季IFRS財報_稅後資產報酬率(%)'
    debt_ratio = '季IFRS財報_負債比率(%)'
    if roaa in df.columns and debt_ratio in df.columns:
        df['ROA_負債交互'] = df[roaa] * df[debt_ratio]

    return df
#df = derive_institutional_financial_features(df)
#df_test = derive_institutional_financial_features(df_test)

def derive_monthly_revenue_features(df):
    """
    衍生月營收相關特徵：
    - 營收成長率的 MA5/MA12（移動平均）
    - 年成長與月變動的交叉點
    - 年營收預估與實際累積的差距
    """
    # 1. 成長率移動平均（技術指標概念）
    if '月營收_單月合併營收年成長(%)' in df.columns:
        df['營收年成長_MA5'] = df['月營收_單月合併營收年成長(%)'].rolling(window=5, min_periods=1).mean()
        df['營收年成長_MA12'] = df['月營收_單月合併營收年成長(%)'].rolling(window=12, min_periods=1).mean()

    # 2. 年成長 + 月變動 交叉（雙成長都為正）
    if all(col in df.columns for col in ['月營收_單月合併營收年成長(%)', '月營收_單月合併營收月變動(%)']):
        # or 70
        df['營收_雙成長交叉'] = (
            (df['月營收_單月合併營收年成長(%)'] > 50) &
            (df['月營收_單月合併營收月變動(%)'] > 50)
        ).astype(int)

    # 3. 估算年營收 vs 累計實際（用12倍月營收近似全年）
    if '月營收_單月合併營收(千)' in df.columns and '月營收_累計合併營收(千)' in df.columns:
        df['營收_年預估與累計差'] = df['月營收_單月合併營收(千)'] * 12 - df['月營收_累計合併營收(千)']

    return df

#df = derive_monthly_revenue_features(df)
#df_test = derive_monthly_revenue_features(df_test)

def derive_chips_features(df):
    """
    衍生個股主力籌碼相關特徵：
    - 買超券商是否重複（集中度指標）
    - 買均價 vs 收盤價乖離
    - 買超張數變動率異常
    """

    # 1. 主力集中度（Top 5 券商是否持續出現）
    top_brokers = [
        '買超第1名分點券商代號',
        '買超第2名分點券商代號',
        '買超第3名分點券商代號',
        '買超第4名分點券商代號',
        '買超第5名分點券商代號',
        '買超第1名分點前1天券商代號',
        '買超第2名分點前1天券商代號',
        '買超第3名分點前1天券商代號',
        '買超第4名分點前1天券商代號',
        '買超第5名分點前1天券商代號'
    ]
    valid_brokers = [col for col in top_brokers if col in df.columns]
    if len(valid_brokers) >= 2:
        df['主力分點重複家數'] = df[valid_brokers].apply(lambda row: len(set(row)), axis=1)

    # 2. 買均價 vs 收盤價乖離（需要收盤價欄位）
    #if '買超第1名分點買均價' in df.columns and '收盤價' in df.columns:
    #    df['主力買均價乖離'] = (df['收盤價'] - df['買超第1名分點買均價']) / (df['買超第1名分點買均價'] + 1e-6)

    # 3. 張數變動率（近1日 vs 5日 vs 10日）
    cols = [
        '個股主力買賣超統計_近1日主力買賣超',
        '個股主力買賣超統計_近5日主力買賣超',
        '個股主力買賣超統計_近10日主力買賣超'
    ]
    if all(c in df.columns for c in cols):
        df['主力買賣變動率_1v5'] = (df[cols[0]] - df[cols[1]]) / (abs(df[cols[1]]) + 1e-6)
        df['主力買賣變動率_1v10'] = (df[cols[0]] - df[cols[2]]) / (abs(df[cols[2]]) + 1e-6)

    return df
#df = derive_chips_features(df)
#df_test = derive_chips_features(df_test)

def add_all_features(df, quantile_labeled=True):
    """
    整合所有特徵工程模組：
    - 技術指標（如黃金交叉、乖離率趨勢、共振）
    - 外資/主力分點行為
    - 法人與財務交互特徵
    - 月營收趨勢與預估差距
    - 主力籌碼異常與集中度

    Args:
        df: 原始 DataFrame
        quantile_labeled: True 若欄位為 qcut 後的等分標籤（0~99）

    Returns:
        df: 加上衍生特徵的 DataFrame
    """
    # --- 技術指標特徵 ---
    df = TI_new(df)

    # --- 外資券商特徵 ---
    df = derive_broker_features(df, n=3)

    # --- 法人與財務交互 ---
    df = derive_institutional_financial_features(df)

    # --- 月營收趨勢與雙重成長邏輯 ---
    df = derive_monthly_revenue_features(df)

    # --- 主力籌碼異常行為與集中度 ---
    df = derive_chips_features(df)

    return df
#df = add_all_features(df, quantile_labeled=True)
#df_test = add_all_features(df_test, quantile_labeled=True)


# --- Utility: Signed log1p transform ---
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))

# --- Enhanced Feature Transformer ---
def enhance_features(df, columns_to_transform):
    for col in columns_to_transform:
        if col not in df.columns:
            continue
        try:
            # Signed log1p version
            df[col + '_signedlog'] = signed_log1p(df[col])

            # qcut to 100 bins
            df[col + '_qcut'] = pd.qcut(df[col], q=100, labels=False, duplicates='drop')
        except Exception:
            df[col + '_qcut'] = np.nan
    return df

def train_lightgbm_cv_f1(df, features, target_col='飆股', n_splits=5):
    X = df[features]
    y = df[target_col]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    f1_scores = []

    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)  # Create 'models' folder if not present

    # Get current date string in YYYYMMDD format for filenames
    date_str = datetime.now().strftime("%Y%m%d")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params={
                'objective': 'binary',
                'metric': 'None',  # Disable internal metrics to use custom eval
                'verbosity': -1
            },
            train_set=dtrain,
            valid_sets=[dtrain, dval],
            num_boost_round=1000
        )

        val_preds = model.predict(X_val)
        f1 = f1_score(y_val, (val_preds > 0.5).astype(int))
        f1_scores.append(f1)

        # Save the trained model for this fold to disk
        filename = f"model-{date_str}-fold-{fold+1}.pkl"
        filepath = os.path.join("models", filename)
        with open(filepath, "wb") as f:
            pickle.dump(model, f)

        models.append(model)
        print(f"Fold {fold+1} F1 Score: {f1:.4f}")

    print(f"Mean F1 Score: {np.mean(f1_scores):.4f}")
    return models

# --- Revised LightGBM Parameters (Optimized) ---
lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_boost_round': 1000,
    'feature_fraction': 0.36341113351903775,
    'lambda_l1': 1.4306966747518972,
    'lambda_l2': 1.1342572678210154,
    'learning_rate': 0.16988128303403846,
    'max_depth': 4,
    'min_data_in_leaf': 11,
    'min_gain_to_split': 0.9807641983846155,
    'min_sum_hessian_in_leaf': 0.006851449088462784,
    'num_leaves': 12,
    'early_stopping_rounds': 50,
    'verbose': -1
}
