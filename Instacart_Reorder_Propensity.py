"""
Instacart EDA for User–Product Reorder Propensity Prediction
- Reads data from a local folder (your provided path)
- Uses chunking for order_products__prior.csv to avoid memory issues
- Produces plots + summary CSVs for feature ideation

Dataset structure/columns follow Instacart Market Basket Analysis references.
(Kaggle/GitHub descriptions) and your internal column mapping.
"""

import os
import sys
from pathlib import Path

import seaborn as sns
print(sns.__version__)

# ----------------------------
# 0) CONFIG
# ----------------------------
DATA_DIR = Path(r"C:\Users\HHSS\OneDrive - Bayer\Downloads\archive (3)")
OUT_DIR = DATA_DIR / "eda_outputs"
PLOTS_DIR = OUT_DIR / "plots"
TABLES_DIR = OUT_DIR / "tables"

# If you want quick iteration first, keep FAST_MODE=True.
# It will sample users/orders so you can validate logic and plots quickly.
FAST_MODE = True
SAMPLE_USERS = 20000          # increase if your machine can handle
RANDOM_STATE = 42

# Chunk size for reading the huge prior file
PRIOR_CHUNKSIZE = 2_000_000   # tune up/down based on RAM

TOP_N = 20

# ----------------------------
# 1) DEPENDENCY CHECKS
# ----------------------------
def _require(pkg_name, import_name=None):
    try:
        __import__(import_name or pkg_name)
        return True
    except ImportError:
        print(f"\nMissing package: {pkg_name}")
        print(f"Install with: pip install {pkg_name}\n")
        return False

ok = True
ok &= _require("pandas")
ok &= _require("numpy")
ok &= _require("matplotlib", "matplotlib")
ok &= _require("seaborn")
if not ok:
    sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")


# ----------------------------
# 2) UTILITIES
# ----------------------------
def ensure_dirs():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

def list_files():
    print("\nData directory contents:")
    for p in sorted(DATA_DIR.glob("*")):
        print(" -", p.name)

def read_csv_safe(path: Path, **kwargs):
    if not path.exists():
        print(f"\nERROR: Missing file: {path}")
        list_files()
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, **kwargs)

def save_plot(fig, filename):
    out = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

def write_table(df, filename):
    out = TABLES_DIR / filename
    df.to_csv(out, index=False)


# ----------------------------
# 3) LOAD SMALL TABLES (safe in memory)
# ----------------------------
def load_base_tables():
    # common Instacart files (as per public dataset descriptions)
    orders = read_csv_safe(
        DATA_DIR / "orders.csv",
        dtype={
            "order_id": np.int32,
            "user_id": np.int32,
            "eval_set": "category",
            "order_number": np.int16,
            "order_dow": np.int8,
            "order_hour_of_day": np.int8,
        }
    )
    # days_since_prior_order has NaN for first order
    orders["days_since_prior_order"] = orders["days_since_prior_order"].astype("float32")

    products = read_csv_safe(
        DATA_DIR / "products.csv",
        dtype={
            "product_id": np.int32,
            "aisle_id": np.int16,
            "department_id": np.int8
        }
    )
    aisles = read_csv_safe(DATA_DIR / "aisles.csv", dtype={"aisle_id": np.int16})
    departments = read_csv_safe(DATA_DIR / "departments.csv", dtype={"department_id": np.int8})

    # train is much smaller than prior -> load directly
    train = read_csv_safe(
        DATA_DIR / "order_products__train.csv",
        dtype={
            "order_id": np.int32,
            "product_id": np.int32,
            "add_to_cart_order": np.int16,
            "reordered": np.int8
        }
    )

    return orders, products, aisles, departments, train


# ----------------------------
# 4) OPTIONAL SAMPLING (FAST_MODE)
# ----------------------------
def apply_sampling(orders, train):
    if not FAST_MODE:
        return orders, train, None

    # sample users from all users present
    rng = np.random.default_rng(RANDOM_STATE)
    users = orders["user_id"].unique()
    sample_users = rng.choice(users, size=min(SAMPLE_USERS, len(users)), replace=False)

    orders_s = orders[orders["user_id"].isin(sample_users)].copy()
    train_s = train[train["order_id"].isin(orders_s.loc[orders_s.eval_set == "train", "order_id"])].copy()

    # We'll filter prior chunks using this set of prior order_ids
    prior_order_ids = set(orders_s.loc[orders_s.eval_set == "prior", "order_id"].astype(np.int32).tolist())

    print(f"\nFAST_MODE enabled:")
    print(f" - sampled users: {len(sample_users):,}")
    print(f" - sampled orders: {len(orders_s):,}")
    print(f" - sampled train rows: {len(train_s):,}")
    print(f" - sampled prior order_ids: {len(prior_order_ids):,}")

    return orders_s, train_s, prior_order_ids


# ----------------------------
# 5) CHUNKED PROCESSING FOR PRIOR
# ----------------------------
def process_prior_in_chunks(orders, products, prior_order_ids=None):
    """
    Streams through order_products__prior.csv and accumulates:
    - overall reorder stats
    - product-level reorder stats
    - aisle/department-level reorder stats
    - order basket sizes
    - time heatmap stats (dow x hour)
    - days_since_prior_order stats
    - user-level stats (optional; based on sampled orders in FAST_MODE)
    """
    prior_path = DATA_DIR / "order_products__prior.csv"
    if not prior_path.exists():
        list_files()
        raise FileNotFoundError(str(prior_path))

    # order lookup maps
    order_to_user = orders.set_index("order_id")["user_id"]
    order_to_num  = orders.set_index("order_id")["order_number"]
    order_to_dow  = orders.set_index("order_id")["order_dow"]
    order_to_hour = orders.set_index("order_id")["order_hour_of_day"]
    order_to_days = orders.set_index("order_id")["days_since_prior_order"]

    # product -> aisle/department maps
    prod_to_aisle = products.set_index("product_id")["aisle_id"]
    prod_to_dept  = products.set_index("product_id")["department_id"]

    # accumulators
    overall_n = 0
    overall_reordered = 0

    # product stats
    prod_cnt = {}
    prod_reo = {}

    # aisle/dept stats
    aisle_cnt = {}
    aisle_reo = {}
    dept_cnt = {}
    dept_reo = {}

    # order basket sizes
    order_item_cnt = {}

    # dow-hour heatmap accumulators: (dow,hour) -> (reordered_sum, total_cnt)
    dh_reo = {}
    dh_cnt = {}

    # days_since_prior_order buckets
    # We'll bucket into integer days 0..30 plus NaN handled
    day_reo = {}
    day_cnt = {}

    # user stats (in FAST_MODE sampled orders, else can be huge)
    user_cnt = {}
    user_reo = {}

    use_user_stats = FAST_MODE  # safe default

    print("\nStreaming prior file in chunks...")

    reader = pd.read_csv(
        prior_path,
        dtype={
            "order_id": np.int32,
            "product_id": np.int32,
            "add_to_cart_order": np.int16,
            "reordered": np.int8
        },
        chunksize=PRIOR_CHUNKSIZE
    )

    for i, chunk in enumerate(reader, start=1):
        if prior_order_ids is not None:
            chunk = chunk[chunk["order_id"].isin(prior_order_ids)]
            if chunk.empty:
                continue

        # attach order-level context (map is faster than merge for chunking)
        chunk["user_id"] = chunk["order_id"].map(order_to_user).astype(np.int32)
        chunk["order_number"] = chunk["order_id"].map(order_to_num).astype(np.int16)
        chunk["order_dow"] = chunk["order_id"].map(order_to_dow).astype(np.int8)
        chunk["order_hour_of_day"] = chunk["order_id"].map(order_to_hour).astype(np.int8)
        chunk["days_since_prior_order"] = chunk["order_id"].map(order_to_days).astype("float32")

        # attach product-level category
        chunk["aisle_id"] = chunk["product_id"].map(prod_to_aisle).astype(np.int16)
        chunk["department_id"] = chunk["product_id"].map(prod_to_dept).astype(np.int8)

        # overall stats
        overall_n += len(chunk)
        overall_reordered += int(chunk["reordered"].sum())

        # product stats
        g = chunk.groupby("product_id")["reordered"].agg(["count", "sum"])
        for pid, row in g.iterrows():
            prod_cnt[pid] = prod_cnt.get(pid, 0) + int(row["count"])
            prod_reo[pid] = prod_reo.get(pid, 0) + int(row["sum"])

        # aisle stats
        g = chunk.groupby("aisle_id")["reordered"].agg(["count", "sum"])
        for aid, row in g.iterrows():
            aisle_cnt[aid] = aisle_cnt.get(aid, 0) + int(row["count"])
            aisle_reo[aid] = aisle_reo.get(aid, 0) + int(row["sum"])

        # dept stats
        g = chunk.groupby("department_id")["reordered"].agg(["count", "sum"])
        for did, row in g.iterrows():
            dept_cnt[did] = dept_cnt.get(did, 0) + int(row["count"])
            dept_reo[did] = dept_reo.get(did, 0) + int(row["sum"])

        # order basket size
        oc = chunk.groupby("order_id")["product_id"].count()
        for oid, c in oc.items():
            order_item_cnt[oid] = order_item_cnt.get(oid, 0) + int(c)

        # dow-hour heatmap
        gh = chunk.groupby(["order_dow", "order_hour_of_day"])["reordered"].agg(["count", "sum"])
        for (dow, hr), row in gh.iterrows():
            key = (int(dow), int(hr))
            dh_cnt[key] = dh_cnt.get(key, 0) + int(row["count"])
            dh_reo[key] = dh_reo.get(key, 0) + int(row["sum"])

        # days_since_prior_order stats (bucket to nearest int day)
        # NaN values (first order) won't appear much in prior, but handle robustly
        tmp_day = chunk["days_since_prior_order"].fillna(-1).round().astype(np.int16)
        gd = chunk.groupby(tmp_day)["reordered"].agg(["count", "sum"])
        for d, row in gd.iterrows():
            dd = int(d)
            day_cnt[dd] = day_cnt.get(dd, 0) + int(row["count"])
            day_reo[dd] = day_reo.get(dd, 0) + int(row["sum"])

        # user stats (FAST_MODE only)
        if use_user_stats:
            gu = chunk.groupby("user_id")["reordered"].agg(["count", "sum"])
            for uid, row in gu.iterrows():
                uid = int(uid)
                user_cnt[uid] = user_cnt.get(uid, 0) + int(row["count"])
                user_reo[uid] = user_reo.get(uid, 0) + int(row["sum"])

        if i % 5 == 0:
            print(f"  processed {i} chunks... (rows so far: {overall_n:,})")

    # Build DataFrames
    overall_reorder_rate = overall_reordered / max(overall_n, 1)

    prod_stats = pd.DataFrame({
        "product_id": list(prod_cnt.keys()),
        "purchases": [prod_cnt[k] for k in prod_cnt.keys()],
        "reorders": [prod_reo.get(k, 0) for k in prod_cnt.keys()],
    })
    prod_stats["reorder_rate"] = prod_stats["reorders"] / prod_stats["purchases"]

    aisle_stats = pd.DataFrame({
        "aisle_id": list(aisle_cnt.keys()),
        "purchases": [aisle_cnt[k] for k in aisle_cnt.keys()],
        "reorders": [aisle_reo.get(k, 0) for k in aisle_cnt.keys()],
    })
    aisle_stats["reorder_rate"] = aisle_stats["reorders"] / aisle_stats["purchases"]

    dept_stats = pd.DataFrame({
        "department_id": list(dept_cnt.keys()),
        "purchases": [dept_cnt[k] for k in dept_cnt.keys()],
        "reorders": [dept_reo.get(k, 0) for k in dept_cnt.keys()],
    })
    dept_stats["reorder_rate"] = dept_stats["reorders"] / dept_stats["purchases"]

    basket = pd.DataFrame({
        "order_id": list(order_item_cnt.keys()),
        "basket_size": [order_item_cnt[k] for k in order_item_cnt.keys()]
    })

    dh = []
    for (dow, hr), c in dh_cnt.items():
        s = dh_reo.get((dow, hr), 0)
        dh.append([dow, hr, c, s, s / max(c, 1)])
    dh_stats = pd.DataFrame(dh, columns=["order_dow", "order_hour_of_day", "count", "reorders", "reorder_rate"])

    day = []
    for d, c in day_cnt.items():
        s = day_reo.get(d, 0)
        day.append([d, c, s, s / max(c, 1)])
    day_stats = pd.DataFrame(day, columns=["days_since_prior_order_int", "count", "reorders", "reorder_rate"])
    day_stats = day_stats.sort_values("days_since_prior_order_int")

    user_stats = None
    if use_user_stats:
        user_stats = pd.DataFrame({
            "user_id": list(user_cnt.keys()),
            "items_purchased": [user_cnt[k] for k in user_cnt.keys()],
            "items_reordered": [user_reo.get(k, 0) for k in user_cnt.keys()],
        })
        user_stats["user_reorder_rate"] = user_stats["items_reordered"] / user_stats["items_purchased"]

    print("\nPrior streaming complete.")
    print(f"Overall prior reorder rate: {overall_reorder_rate:.4f}")

    return overall_reorder_rate, prod_stats, aisle_stats, dept_stats, basket, dh_stats, day_stats, user_stats


# ----------------------------
# 6) TRAIN SET QUICK EDA (labels)
# ----------------------------
def train_label_eda(orders, train, products):
    # attach order context
    train2 = train.merge(
        orders[["order_id", "user_id", "order_number", "order_dow", "order_hour_of_day", "days_since_prior_order"]],
        on="order_id",
        how="left"
    )
    # attach product category
    train2 = train2.merge(products[["product_id", "aisle_id", "department_id"]], on="product_id", how="left")

    label_rate = train2["reordered"].mean()
    print(f"\nTRAIN label reorder rate (mean reordered): {label_rate:.4f}")

    return train2, label_rate


# ----------------------------
# 7) PLOTS
# ----------------------------
def make_plots(orders, products, aisles, departments,
               overall_reorder_rate, prod_stats, aisle_stats, dept_stats, basket, dh_stats, day_stats,
               user_stats, train2):

    # merge names for readability
    prod_named = prod_stats.merge(products[["product_id", "product_name", "aisle_id", "department_id"]],
                                  on="product_id", how="left")
    prod_named = prod_named.merge(aisles, on="aisle_id", how="left").merge(departments, on="department_id", how="left")

    aisle_named = aisle_stats.merge(aisles, on="aisle_id", how="left")
    dept_named = dept_stats.merge(departments, on="department_id", how="left")

    # 1) Orders per user distribution
    orders_per_user = orders.groupby("user_id")["order_id"].count().reset_index(name="n_orders")
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(orders_per_user["n_orders"], bins=50, kde=False)
    plt.title("Distribution: Number of Orders per User")
    plt.xlabel("Orders per user")
    plt.ylabel("Count of users")
    save_plot(fig, "orders_per_user_hist.png")
    write_table(orders_per_user, "orders_per_user.csv")

    # 2) Basket size distribution
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(basket["basket_size"], bins=60, kde=False)
    plt.title("Distribution: Basket Size (# items per order)")
    plt.xlabel("Basket size")
    plt.ylabel("Count of orders")
    save_plot(fig, "basket_size_hist.png")
    write_table(basket, "basket_size_by_order.csv")

    # 3) Heatmap: reorder rate by day-of-week x hour
    pivot = dh_stats.pivot(index="order_dow", columns="order_hour_of_day", values="reorder_rate")
    fig = plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, cmap="viridis", cbar_kws={"label": "Reorder rate"})
    plt.title("Reorder Rate Heatmap: Day-of-Week x Hour-of-Day")
    plt.xlabel("Hour of day")
    plt.ylabel("Day of week (0=Sun per Instacart convention)")
    save_plot(fig, "reorder_rate_heatmap_dow_hour.png")
    write_table(dh_stats, "reorder_rate_by_dow_hour.csv")

    # 4) Reorder rate vs days_since_prior_order
    # exclude -1 bucket (if present)
    day_plot = day_stats[day_stats["days_since_prior_order_int"] >= 0].copy()
    fig = plt.figure(figsize=(12, 6))
    sns.lineplot(data=day_plot, x="days_since_prior_order_int", y="reorder_rate", marker="o")
    plt.title("Reorder Rate vs Days Since Prior Order")
    plt.xlabel("Days since prior order (rounded)")
    plt.ylabel("Reorder rate")
    save_plot(fig, "reorder_rate_vs_days_since.png")
    write_table(day_plot, "reorder_rate_by_days_since_prior.csv")

    # 5) Top products by purchases
    top_p = prod_named.sort_values("purchases", ascending=False).head(TOP_N)
    fig = plt.figure(figsize=(12, 8))
    sns.barplot(data=top_p, y="product_name", x="purchases")
    plt.title(f"Top {TOP_N} Products by Purchases (PRIOR)")
    plt.xlabel("Purchases")
    plt.ylabel("")
    save_plot(fig, "top_products_by_purchases.png")
    write_table(top_p, "top_products_by_purchases.csv")

    # 6) Top products by reorder rate (with minimum purchases filter)
    min_p = max(200, int(prod_named["purchases"].quantile(0.90) / 10))  # dynamic threshold
    top_rr = (prod_named[prod_named["purchases"] >= min_p]
              .sort_values("reorder_rate", ascending=False)
              .head(TOP_N))
    fig = plt.figure(figsize=(12, 8))
    sns.barplot(data=top_rr, y="product_name", x="reorder_rate")
    plt.title(f"Top {TOP_N} Products by Reorder Rate (min purchases ≥ {min_p})")
    plt.xlabel("Reorder rate")
    plt.ylabel("")
    save_plot(fig, "top_products_by_reorder_rate.png")
    write_table(top_rr, "top_products_by_reorder_rate.csv")

    # 7) Aisle reorder rate (top aisles by purchases)
    top_ais = aisle_named.sort_values("purchases", ascending=False).head(TOP_N)
    fig = plt.figure(figsize=(12, 8))
    sns.barplot(data=top_ais, y="aisle", x="reorder_rate")
    plt.title(f"Top {TOP_N} Aisles by Purchases: Reorder Rate")
    plt.xlabel("Reorder rate")
    plt.ylabel("")
    save_plot(fig, "aisle_reorder_rate_top_purchases.png")
    write_table(top_ais, "aisle_stats_top_purchases.csv")

    # 8) Department reorder rate (all depts)
    fig = plt.figure(figsize=(10, 6))
    dept_sorted = dept_named.sort_values("reorder_rate", ascending=False)
    sns.barplot(data=dept_sorted, y="department", x="reorder_rate")
    plt.title("Department Reorder Rates")
    plt.xlabel("Reorder rate")
    plt.ylabel("")
    save_plot(fig, "department_reorder_rates.png")
    write_table(dept_sorted, "department_stats.csv")

    # 9) Train label rate by order_number (does reorder increase with user maturity?)
    fig = plt.figure(figsize=(12, 6))
    tmp = train2.groupby("order_number")["reordered"].mean().reset_index()
    sns.lineplot(data=tmp, x="order_number", y="reordered")
    plt.title("TRAIN: Reordered Rate vs Order Number")
    plt.xlabel("Order number")
    plt.ylabel("Mean reordered")
    save_plot(fig, "train_reorder_rate_vs_order_number.png")
    write_table(tmp, "train_reorder_rate_by_order_number.csv")

    # 10) User-level reorder distribution (FAST_MODE only)
    if user_stats is not None:
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(user_stats["user_reorder_rate"], bins=50, kde=True)
        plt.title("User Reorder Rate Distribution (sampled users)")
        plt.xlabel("User reorder rate")
        plt.ylabel("Count of users")
        save_plot(fig, "user_reorder_rate_hist.png")
        write_table(user_stats.sort_values("user_reorder_rate", ascending=False), "user_stats.csv")

    # Save core stats
    write_table(prod_stats.sort_values("purchases", ascending=False), "product_stats.csv")
    write_table(aisle_stats.sort_values("purchases", ascending=False), "aisle_stats.csv")
    write_table(dept_stats.sort_values("purchases", ascending=False), "department_stats_raw.csv")

    print("\nPlots and tables saved to:", OUT_DIR)


# ----------------------------
# 8) OPTIONAL: USER–PRODUCT AGGREGATES (FAST_MODE)
# ----------------------------
def build_user_product_features(orders, train2, products):
    """
    Creates sample user-product features from TRAIN only (safe and small),
    plus product priors (from prod_stats output) for feature ideation.

    For full user-product across PRIOR (32M rows), you typically build this in
    a modeling pipeline with chunked aggregation; EDA uses sampling.
    """
    # Basic user-product frequency features from TRAIN
    up = (train2.groupby(["user_id", "product_id"])
          .agg(
              times_in_train=("order_id", "count"),
              reordered_in_train=("reordered", "sum"),
              mean_cart_pos=("add_to_cart_order", "mean"),
              last_order_number=("order_number", "max"),
          )
          .reset_index())

    up["train_reorder_rate_user_product"] = up["reordered_in_train"] / up["times_in_train"]

    # attach product metadata
    up = up.merge(products[["product_id", "aisle_id", "department_id"]], on="product_id", how="left")

    # Save only a manageable sample
    if len(up) > 500_000:
        up = up.sample(500_000, random_state=RANDOM_STATE)

    write_table(up, "user_product_features_sample.csv")
    print("Saved user-product feature sample: user_product_features_sample.csv")


# ----------------------------
# 9) MAIN
# ----------------------------
def main():
    ensure_dirs()

    print("Loading base tables...")
    orders, products, aisles, departments, train = load_base_tables()

    # sanity print
    print("\nBase table sizes:")
    print("orders:", len(orders))
    print("products:", len(products))
    print("aisles:", len(aisles))
    print("departments:", len(departments))
    print("train:", len(train))

    # sampling for FAST_MODE
    orders_s, train_s, prior_order_ids = apply_sampling(orders, train)

    # process prior with chunking
    overall_rr, prod_stats, aisle_stats, dept_stats, basket, dh_stats, day_stats, user_stats = process_prior_in_chunks(
        orders_s, products, prior_order_ids=prior_order_ids
    )

    # train label EDA
    train2, train_label_rate = train_label_eda(orders_s, train_s, products)

    # plots
    make_plots(
        orders_s, products, aisles, departments,
        overall_rr, prod_stats, aisle_stats, dept_stats, basket, dh_stats, day_stats,
        user_stats, train2
    )

    # optional user-product sample features
    build_user_product_features(orders_s, train2, products)

    # concise EDA summary
    summary = {
        "n_users": int(orders_s["user_id"].nunique()),
        "n_orders": int(orders_s["order_id"].nunique()),
        "n_products": int(products["product_id"].nunique()),
        "prior_overall_reorder_rate": float(overall_rr),
        "train_label_reorder_rate": float(train_label_rate),
        "fast_mode": bool(FAST_MODE)
    }
    pd.DataFrame([summary]).to_csv(TABLES_DIR / "eda_summary.csv", index=False)
    print("\nEDA summary saved: eda_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()