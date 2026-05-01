import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import holidays
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Data Preparation Functions
# ------------------------------------------------------------------------------


def get_holiday_tensor(dates, device):
    us_holidays = holidays.US()  # or holidays.US(state="NY")

    # DatetimeIndex -> list of python date objects
    holiday_dates = {d for d in (dt.date() for dt in dates) if d in us_holidays}

    is_holiday = pd.Index([dt.date() for dt in dates]).isin(holiday_dates)

    H_tensor = torch.tensor(is_holiday.astype(float), dtype=torch.float32).unsqueeze(1).to(device)
    print(f"Holiday tensor shape: {H_tensor.shape}. Found {int(H_tensor.sum().item())} holidays.")
    return H_tensor



def reshape_for_tcp(X_df,
                    feature_cols=None,
                    target_col="complaint_count"):
    """
    Convert feature matrix (region_id, date, columns...) into
    TCP-style tensors and generate day-type masks.

    Returns:
    X_tensor : ndarray of shape (K, N, M)
    Y        : ndarray of shape (K, N)
    regions  : ndarray of region_ids of length N
    dates    : DatetimeIndex of length K
    masks    : dict containing boolean numpy arrays for 'wd', 'fri', 'we'
    """

    df = X_df.copy()

    # Ensure proper dtypes and sorting
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "region_id"])

    # Unique regions and dates in fixed order
    regions = np.sort(df["region_id"].unique())
    dates = pd.Index(sorted(df["date"].unique()))

    N = len(regions)
    K = len(dates)

    # Pivot logic
    def pivot_col(col):
        table = (
            df.pivot(index="date", columns="region_id", values=col)
            .reindex(index=dates, columns=regions)
        )
        return table.values.astype(float)  # shape (K, N)

    # Target tensor (K, N)
    Y = pivot_col(target_col)

    # Stack feature tensors along last axis -> (K, N, M)
    if feature_cols:
        feat_arrays = [pivot_col(col)[:, :, None] for col in feature_cols]
        X_tensor = np.concatenate(feat_arrays, axis=-1)
    else:
        X_tensor = np.zeros((K, N, 0))

    print(f"X_tensor shape: {X_tensor.shape}  (K, N, M)")
    print(f"Y shape:        {Y.shape}        (K, N)")

    # --- Generate Day-Type Masks ---
    # 0=Monday, 6=Sunday
    dayofweek = dates.dayofweek.values

    masks = {
        "wd": (dayofweek >= 0) & (dayofweek <= 3),  # Mon-Thu
        "fri": (dayofweek == 4),  # Fri
        "we": (dayofweek >= 5)  # Sat-Sun
    }

    return X_tensor, Y, regions, dates, masks


def build_neighbor_pairs(grid_active, regions):
    """
    Build list of neighbor index pairs from grid polygons.
    """
    # Ensure GeoDataFrame is in same order as regions
    grid = grid_active.set_index("region_id").loc[regions].reset_index()

    sindex = grid.sindex
    neighbor_pairs = set()

    for i, geom in enumerate(grid.geometry):
        possible = list(sindex.intersection(geom.bounds))
        for j in possible:
            if i == j:
                continue
            if geom.touches(grid.geometry.iloc[j]):
                a, b = sorted((i, j))
                neighbor_pairs.add((a, b))

    neighbor_pairs = sorted(neighbor_pairs)
    print(f"Found {len(neighbor_pairs)} neighbor pairs")
    return neighbor_pairs


def process_neighbor_indices(neighbor_pairs, device):
    """
    Helper to convert list of tuples [(i, j), ...] into two tensors
    for vectorized indexing.
    """
    if not neighbor_pairs:
        return None, None

    idx_i = [p[0] for p in neighbor_pairs]
    idx_j = [p[1] for p in neighbor_pairs]

    return torch.tensor(idx_i, device=device), torch.tensor(idx_j, device=device)


def compute_armse(y_true, y_pred):
    """
    Compute average RMSE across regions (aRMSE).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape for aRMSE.")
    if y_true.ndim != 2:
        raise ValueError("aRMSE expects 2D arrays shaped (K, N).")
    rmse_per_region = np.sqrt(np.nanmean((y_pred - y_true) ** 2, axis=0))
    return float(np.nanmean(rmse_per_region))


def build_rolling_splits(K, train_size, val_size, test_size, step_size,
                         device=None, drop_last=True):
    """
    Create rolling (walk-forward) train/val/test splits over time indices.

    Example (K=365, train=255, val=55, test=55, step=28) will yield
    multiple splits that move forward by step_size each time.
    """
    if any(s <= 0 for s in [train_size, val_size, test_size, step_size]):
        raise ValueError("train_size, val_size, test_size, and step_size must be > 0.")
    if K < train_size + val_size + test_size:
        raise ValueError("Not enough time steps for the requested split sizes.")

    splits = []
    start = 0
    max_start = K - (train_size + val_size + test_size)
    while start <= max_start:
        train_start = start
        train_end = train_start + train_size
        val_end = train_end + val_size
        test_end = val_end + test_size

        train_idx = torch.arange(train_start, train_end, device=device)
        val_idx = torch.arange(train_end, val_end, device=device)
        test_idx = torch.arange(val_end, test_end, device=device)

        splits.append({
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "train_val_idx": torch.arange(train_start, val_end, device=device),
        })
        start += step_size

    if not splits and not drop_last:
        train_start = 0
        train_end = min(K, train_size)
        val_end = min(K, train_end + val_size)
        test_end = min(K, val_end + test_size)
        splits.append({
            "train_idx": torch.arange(train_start, train_end, device=device),
            "val_idx": torch.arange(train_end, val_end, device=device),
            "test_idx": torch.arange(val_end, test_end, device=device),
            "train_val_idx": torch.arange(train_start, val_end, device=device),
        })

    return splits


# ------------------------------------------------------------------------------
# Sliding-Window Alpha Prediction Helpers
# ------------------------------------------------------------------------------

def _fit_alpha_for_region(W_history, window_size=7, ridge=1e-4):
    """
    Learn alpha coefficients for a single region by solving Eq. (20) in the TCP paper.
    W_history: numpy array of shape (T_hist, M)
    Returns: (alpha vector, effective_window_size)
    """
    window_size = max(1, int(window_size))
    T_hist, M = W_history.shape

    if T_hist == 0:
        return np.array([1.0]), 1
    if T_hist == 1:
        return np.array([1.0]), 1

    g = min(window_size, T_hist - 1)
    num_samples = (T_hist - g) * M
    if num_samples <= 0:
        alpha = np.ones(g) / g
        return alpha, g

    rows = []
    targets = []
    for idx in range(g, T_hist):
        past = W_history[idx - g: idx]  # shape (g, M)
        rows.append(past.T)             # (M, g)
        targets.append(W_history[idx])  # (M,)

    X = np.concatenate(rows, axis=0)    # (#samples, g)
    y = np.concatenate(targets, axis=0) # (#samples,)
    XtX = X.T @ X + ridge * np.eye(g)
    Xty = X.T @ y
    alpha = np.linalg.solve(XtX, Xty)
    return alpha, g


def _forecast_region_weights(W_history, alpha, horizon):
    """
    Generate future W vectors for a region using learned alpha coefficients.
    """
    g = len(alpha)
    if horizon <= 0:
        return np.zeros((0, W_history.shape[1]))

    history = W_history[-g:].copy()
    if history.shape[0] < g:
        pad = np.repeat(history[:1], g - history.shape[0], axis=0)
        history = np.concatenate([pad, history], axis=0)

    preds = []
    for _ in range(horizon):
        new_w = np.tensordot(alpha, history, axes=(0, 0))
        preds.append(new_w)
        if g > 1:
            history = np.concatenate([history[1:], new_w[None, :]], axis=0)
        else:
            history[0] = new_w
    return np.stack(preds, axis=0)


def forecast_weights_with_alpha(W_history, horizon, window_size=7, ridge=1e-4):
    """
    Learn per-region alpha coefficients from historical W values and forecast
    future weights for a contiguous horizon.
    W_history: numpy array (T_hist, N, M)
    Returns:
        forecasts: numpy array (horizon, N, M)
        alphas: list of numpy arrays, one per region
    """
    if horizon <= 0:
        return np.zeros((0, ) + W_history.shape[1:]), []

    T_hist, N, _ = W_history.shape
    forecasts = np.zeros((horizon, N, W_history.shape[2]))
    alphas = []
    for n in range(N):
        region_hist = W_history[:, n, :]
        alpha, g = _fit_alpha_for_region(region_hist, window_size=window_size, ridge=ridge)
        # Ensure history has enough length for the chosen window
        if region_hist.shape[0] < g:
            if region_hist.shape[0] == 0:
                seed = np.zeros((1, W_history.shape[2]))
            else:
                seed = region_hist[-1:]
            effective_hist = np.repeat(seed, g, axis=0)
        else:
            effective_hist = region_hist
        region_forecast = _forecast_region_weights(effective_hist, alpha, horizon)
        forecasts[:, n, :] = region_forecast
        alphas.append(alpha)
    return forecasts, alphas


# ------------------------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------------------------
class TCPBaseline(torch.nn.Module):
    def __init__(self, K, N, M):
        super().__init__()
        # Time-Region-Feature weights
        self.W = torch.nn.Parameter(torch.zeros(K, N, M))

    def forward(self, X):
        """
        X: (K, N, M)
        returns y_hat: (K, N)
        """
        return (X * self.W).sum(dim=-1)

class TCPModel(torch.nn.Module):
    def __init__(self, K, N, M):
        super().__init__()
        # W: Base weights mapping features to crime counts (Time, Region, Feature)
        self.W = torch.nn.Parameter(torch.zeros(K, N, M))

        # B: Holiday offset weights (Holiday_Type, Feature)
        # Assuming binary holiday type (is_holiday or not), so dim is (1, M)
        self.B = torch.nn.Parameter(torch.zeros(1, M))

    def forward(self, X, H):
        """
        X: shape (K, N, M) Features
        H: shape (K, 1)    Holiday indicators
        Returns predictions y_hat: shape (K, N)
        """
        # Calculate effective weights: W_eff = W + (H * B)
        # H @ B -> (K, 1) @ (1, M) -> (K, M)
        # Unsqueeze to broadcast over N regions -> (K, 1, M)
        holiday_offset = (H @ self.B).unsqueeze(1)

        W_eff = self.W + holiday_offset

        # Elementwise multiply and sum over feature dimension
        y_hat = (X * W_eff).sum(dim=-1)
        return y_hat


# ------------------------------------------------------------------------------
# Loss Functions
# ------------------------------------------------------------------------------

def tcp_loss_baseline(model, X, Y, train_idx, neighbor_indices,
                      lam=1.0, mu=1.0):
    """
    Baseline loss:
      1. MSE on training days
      2. Temporal smoothness on W
      3. Spatial smoothness on W (single mu)
    """
    # 1. Data fit on training window
    # Compute predictions only for training indices (avoid full-K forward).
    X_train = X[train_idx]
    W_train = model.W[train_idx]
    y_hat_train = (X_train * W_train).sum(dim=-1)
    Y_train = Y[train_idx]
    data_loss = torch.mean((y_hat_train - Y_train) ** 2)

    # 2. Temporal smoothness
    W_t = W_train[1:, :, :]
    W_tm1 = W_train[:-1, :, :]
    temp_loss = torch.mean((W_t - W_tm1) ** 2)

    # 3. Spatial smoothness (all days treated the same)
    idx_i, idx_j = neighbor_indices
    if idx_i is not None:
        diff_sq = (W_train[:, idx_i, :] - W_train[:, idx_j, :]) ** 2
        spat_loss = diff_sq.mean()
    else:
        spat_loss = torch.tensor(0.0, device=W_train.device)

    loss = data_loss + lam * temp_loss + mu * spat_loss
    parts = {
        "data": data_loss.item(),
        "temp": temp_loss.item(),
        "spat": spat_loss.item()
    }
    return loss, parts


def tcp_loss_extended(model, X, Y, H, train_idx, neighbor_indices, masks,
                      lam=1.0, lam7=0.0, mu_wd=1.0, mu_fri=1.0, mu_we=1.0,
                      gamma=1e-3):
    """
    Compute TCP loss with:
    1. MSE Data Fit
    2. Temporal Smoothness (on W)
    3. Regime-specific Spatial Smoothness (on W)
    4. Weekly temporal coupling (optional)
    5. Ridge Penalty on B (Holiday weights)
    """

    # Slice to training indices up front (avoid full-K forward and full-K regularizers).
    X_train = X[train_idx]
    Y_train = Y[train_idx]
    H_train = H[train_idx]
    W_train = model.W[train_idx]

    # Compute predictions only for training indices.
    holiday_offset_train = (H_train @ model.B).unsqueeze(1)  # (K_train, 1, M)
    W_eff_train = W_train + holiday_offset_train
    y_hat_train = (X_train * W_eff_train).sum(dim=-1)

    # 1. Data fit loss (MSE) on training set
    data_loss = torch.mean((y_hat_train - Y_train) ** 2)

    # 2. Temporal smoothness: L2 norm of (W[t] - W[t-1])
    # We only smooth the base weights W, not the offsets
    W_t = W_train[1:, :, :]
    W_tm1 = W_train[:-1, :, :]
    temp_loss = torch.mean((W_t - W_tm1) ** 2)

    # 2b. Weekly temporal coupling: day t vs t-7
    if W_train.shape[0] > 7:
        W_t7  = W_train[7:, :, :]
        W_tm7 = W_train[:-7, :, :]
        weekly_loss = torch.mean((W_t7 - W_tm7) ** 2)
    else:
        weekly_loss = torch.tensor(0.0, device=W_train.device)


    # 3. Spatial smoothness (Regime-specific)
    loss_spat_wd = torch.tensor(0.0, device=W_train.device)
    loss_spat_fri = torch.tensor(0.0, device=W_train.device)
    loss_spat_we = torch.tensor(0.0, device=W_train.device)

    idx_i, idx_j = neighbor_indices

    if idx_i is not None:
        # Vectorized calculation
        diff_sq = (W_train[:, idx_i, :] - W_train[:, idx_j, :]) ** 2

        # Masks may be numpy arrays (from reshape_for_tcp) or torch tensors.
        if isinstance(masks.get("wd"), torch.Tensor):
            masks_train = {k: masks[k][train_idx] for k in ("wd", "fri", "we")}
        else:
            train_idx_np = train_idx.detach().cpu().numpy() if isinstance(train_idx, torch.Tensor) else np.asarray(train_idx)
            masks_train = {k: masks[k][train_idx_np] for k in ("wd", "fri", "we")}

        if masks_train["wd"].any():
            loss_spat_wd = diff_sq[masks_train["wd"]].mean()
        if masks_train["fri"].any():
            loss_spat_fri = diff_sq[masks_train["fri"]].mean()
        if masks_train["we"].any():
            loss_spat_we = diff_sq[masks_train["we"]].mean()

    # 4. Ridge penalty on holiday offsets
    ridge_b = torch.mean(model.B ** 2)

    # Total Loss
    loss = data_loss + \
           lam * temp_loss + \
           lam7 * weekly_loss + \
           mu_wd * loss_spat_wd + \
           mu_fri * loss_spat_fri + \
           mu_we * loss_spat_we + \
           gamma * ridge_b

    return loss, {
        "data": data_loss.item(),
        "temp": temp_loss.item(),
        "weekly": weekly_loss.item(),
        "spat_wd": loss_spat_wd.item(),
        "spat_fri": loss_spat_fri.item(),
        "spat_we": loss_spat_we.item(),
        "ridge_b": ridge_b.item()
    }

# ------------------------------------------------------------------------------
# Training Functions
# ------------------------------------------------------------------------------

def eval_with_alpha_prediction(model, X_torch, Y_torch, train_idx, eval_idx,
                               window_size=7, alpha_ridge=1e-4, H_torch=None,
                               use_holiday_offset=True):
    """
    Evaluate a model by forecasting W tensors with the sliding-window alpha stage.
    """
    eval_len = int(eval_idx.shape[0])
    if eval_len == 0:
        return 0.0, np.array([]), np.array([])

    with torch.no_grad():
        device = X_torch.device
        dtype = X_torch.dtype

        W_history = model.W.detach()[train_idx].detach().cpu().numpy()
        horizon = eval_len

        weight_forecasts, _ = forecast_weights_with_alpha(
            W_history,
            horizon,
            window_size=window_size,
            ridge=alpha_ridge
        )

        W_forecast = torch.as_tensor(weight_forecasts, device=device, dtype=dtype)
        X_eval = X_torch[eval_idx]

        W_eff = W_forecast
        if use_holiday_offset and hasattr(model, "B") and (H_torch is not None):
            H_eval = H_torch[eval_idx]
            holiday_offset = (H_eval @ model.B).unsqueeze(1)  # (horizon, 1, M)
            W_eff = W_eff + holiday_offset                    # broadcast over N

        y_hat_eval = (X_eval * W_eff).sum(dim=-1)
        y_true_eval = Y_torch[eval_idx]

    y_pred_2d = y_hat_eval.detach().cpu().numpy()
    y_true_2d = y_true_eval.detach().cpu().numpy()
    armse = compute_armse(y_true_2d, y_pred_2d)
    return armse, y_true_2d.ravel(), y_pred_2d.ravel()


def train_baseline(X_torch, Y_torch, train_idx, neighbor_indices,
                   lam, mu, lr=1e-2, num_epochs=200,
                   device="cpu", verbose=False,
                   val_idx=None, early_stopping=False,
                   early_stop_patience=20, early_stop_min_delta=0.0,
                   early_stop_eval_every=5,
                   alpha_window=7, alpha_ridge=1e-4):
    K, N, M = X_torch.shape
    model = TCPBaseline(K, N, M).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_rmse = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        opt.zero_grad()
        loss, parts = tcp_loss_baseline(
            model, X_torch, Y_torch, train_idx, neighbor_indices,
            lam=lam, mu=mu
        )
        loss.backward()
        opt.step()

        if early_stopping and val_idx is not None and (epoch % early_stop_eval_every == 0):
            rmse, _, _ = eval_baseline(
                model, X_torch, Y_torch, train_idx, val_idx,
                alpha_window=alpha_window,
                alpha_ridge=alpha_ridge
            )
            if rmse + early_stop_min_delta < best_rmse:
                best_rmse = rmse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    if verbose:
                        print(f"[Baseline] Early stopping at epoch {epoch} (best aRMSE={best_rmse:.4f})")
                    break

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"[Baseline] Epoch {epoch:4d}  "
                  f"Loss={loss.item():.4f}  "
                  f"Data={parts['data']:.4f}  "
                  f"Temp={parts['temp']:.4f}  "
                  f"Spat={parts['spat']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model

def eval_baseline(model, X_torch, Y_torch, train_idx, eval_idx,
                  alpha_window=7, alpha_ridge=1e-4):
    return eval_with_alpha_prediction(
        model, X_torch, Y_torch,
        train_idx, eval_idx,
        window_size=alpha_window,
        alpha_ridge=alpha_ridge
    )


def grid_search_baseline(X_torch, Y_torch, train_idx, val_idx, neighbor_indices,
                         lam_values, mu_values,
                         lr=1e-2, num_epochs=200, device="cpu",
                         viz_path="baseline_grid_search.png",
                         alpha_window=7, alpha_ridge=1e-4,
                         early_stopping=False, early_stop_patience=20,
                         early_stop_min_delta=0.0, early_stop_eval_every=5):
    """
    Run a brute-force grid search over lam/mu for the baseline model.
    Uses the validation split for model selection and produces a visualization
    summarizing validation RMSE across the grid.
    """
    device = torch.device(device)
    total_configs = len(lam_values) * len(mu_values)
    best_result = None
    best_model = None
    history = []

    with tqdm(total=total_configs, desc="Baseline Grid Search") as pbar:
        for lam in lam_values:
            for mu in mu_values:
                model = train_baseline(
                    X_torch, Y_torch, train_idx, neighbor_indices,
                    lam=lam, mu=mu,
                    lr=lr, num_epochs=num_epochs, device=device,
                    verbose=False,
                    val_idx=val_idx, early_stopping=early_stopping,
                    early_stop_patience=early_stop_patience,
                    early_stop_min_delta=early_stop_min_delta,
                    early_stop_eval_every=early_stop_eval_every,
                    alpha_window=alpha_window,
                    alpha_ridge=alpha_ridge
                )
                rmse, y_true, y_pred = eval_baseline(
                    model, X_torch, Y_torch, train_idx, val_idx,
                    alpha_window=alpha_window,
                    alpha_ridge=alpha_ridge
                )
                history.append({"lam": lam, "mu": mu, "rmse": rmse})

                if best_result is None or rmse < best_result["rmse"]:
                    best_result = {
                        "rmse": rmse,
                        "params": {"lam": lam, "mu": mu},
                        "y_true": y_true,
                        "y_pred": y_pred,
                    }
                    best_model = model
                    pbar.set_postfix({"best_rmse": f"{rmse:.4f}"})
                else:
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                pbar.update(1)

    if best_result is None:
        raise RuntimeError("Grid search did not evaluate any configurations.")

    if viz_path:
        visualize_baseline_grid(history, viz_path)
    print(
        "[Baseline Grid Search] Best Validation aRMSE="
        f"{best_result['rmse']:.4f} with "
        f"lam={best_result['params']['lam']}, "
        f"mu={best_result['params']['mu']}"
    )

    best_result["model"] = best_model
    best_result["history"] = history
    return best_result

def grid_search_extended(X_torch, Y_torch, H_torch, train_idx, val_idx,
                         neighbor_indices, masks_torch,
                         lam,
                         lam7_values, mu_wd_values, mu_fri_values, mu_we_values, gamma_values,
                         lam_values=None,
                         lr=1e-2, num_epochs=200, device="cpu",
                         viz_path="extended_grid_search.png",
                         alpha_window=7, alpha_ridge=1e-4,
                         early_stopping=False, early_stop_patience=20,
                         early_stop_min_delta=0.0, early_stop_eval_every=5):
    """
    Grid search over the extended-model regularizers that are not shared with the baseline.
    Optionally retunes lam (baseline temporal smoothness) jointly with extension parameters.
    """
    device = torch.device(device)
    lam_candidates = lam_values if lam_values is not None else [lam]
    total_configs = (len(lam_candidates) *
                     len(mu_wd_values) * len(mu_fri_values) *
                     len(mu_we_values) * len(gamma_values) * len(lam7_values))
    best_result = None
    best_model = None
    history = []

    with tqdm(total=total_configs, desc="Extended Grid Search") as pbar:
        for lam_candidate in lam_candidates:
            for lam7 in lam7_values:
                for mu_wd in mu_wd_values:
                    for mu_fri in mu_fri_values:
                        for mu_we in mu_we_values:
                            for gamma in gamma_values:
                                model = train_extended(
                                    X_torch, Y_torch, H_torch, train_idx,
                                    neighbor_indices, masks_torch,
                                    lam=lam_candidate, lam7=lam7, mu_wd=mu_wd,
                                    mu_fri=mu_fri, mu_we=mu_we,
                                    gamma=gamma,
                                    lr=lr, num_epochs=num_epochs, device=device,
                                    verbose=False,
                                    val_idx=val_idx, early_stopping=early_stopping,
                                    early_stop_patience=early_stop_patience,
                                    early_stop_min_delta=early_stop_min_delta,
                                    early_stop_eval_every=early_stop_eval_every,
                                    alpha_window=alpha_window,
                                    alpha_ridge=alpha_ridge
                                )
                                rmse, y_true, y_pred = eval_extended(
                                    model, X_torch, Y_torch, H_torch, train_idx, val_idx,
                                    alpha_window=alpha_window,
                                    alpha_ridge=alpha_ridge
                                )
                                history.append({
                                    "lam": lam_candidate,
                                    "lam7": lam7,
                                    "mu_wd": mu_wd,
                                    "mu_fri": mu_fri,
                                    "mu_we": mu_we,
                                    "gamma": gamma,
                                    "rmse": rmse
                                })

                                if best_result is None or rmse < best_result["rmse"]:
                                    best_result = {
                                        "rmse": rmse,
                                        "params": {
                                            "lam": lam_candidate,
                                            "lam7": lam7,
                                            "mu_wd": mu_wd,
                                            "mu_fri": mu_fri,
                                            "mu_we": mu_we,
                                            "gamma": gamma
                                        },
                                        "y_true": y_true,
                                        "y_pred": y_pred,
                                    }
                                    best_model = model
                                    pbar.set_postfix({"best_rmse": f"{rmse:.4f}"})
                                else:
                                    del model
                                    if device.type == "cuda":
                                        torch.cuda.empty_cache()
                                pbar.update(1)

    if best_result is None:
        raise RuntimeError("Extended grid search did not evaluate any configurations.")

    if viz_path:
        visualize_extended_grid(history, viz_path)
    params = best_result["params"]
    print(
        "[Extended Grid Search] Best Validation aRMSE="
        f"{best_result['rmse']:.4f} with "
        f"lam={params.get('lam', lam)}, "
        f"lam7={params['lam7']}, mu_wd={params['mu_wd']}, "
        f"mu_fri={params['mu_fri']}, mu_we={params['mu_we']}, "
        f"gamma={params['gamma']}"
    )

    best_result["model"] = best_model
    best_result["history"] = history
    return best_result

def train_extended(X_torch, Y_torch, H_torch, train_idx,
                   neighbor_indices, masks_torch,
                   lam, lam7, mu_wd, mu_fri, mu_we, gamma,
                   lr=1e-2, num_epochs=200, device="cpu", verbose=False,
                   val_idx=None, early_stopping=False,
                   early_stop_patience=20, early_stop_min_delta=0.0,
                   early_stop_eval_every=5,
                   alpha_window=7, alpha_ridge=1e-4):
    K, N, M = X_torch.shape
    model = TCPModel(K, N, M).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_rmse = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        opt.zero_grad()
        loss, parts = tcp_loss_extended(
            model, X_torch, Y_torch, H_torch, train_idx,
            neighbor_indices, masks_torch,
            lam=lam,lam7=lam7, mu_wd=mu_wd, mu_fri=mu_fri, mu_we=mu_we,
            gamma=gamma
        )
        loss.backward()
        opt.step()

        if early_stopping and val_idx is not None and (epoch % early_stop_eval_every == 0):
            rmse, _, _ = eval_extended(
                model, X_torch, Y_torch, H_torch, train_idx, val_idx,
                alpha_window=alpha_window,
                alpha_ridge=alpha_ridge
            )
            if rmse + early_stop_min_delta < best_rmse:
                best_rmse = rmse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    if verbose:
                        print(f"[Extended] Early stopping at epoch {epoch} (best aRMSE={best_rmse:.4f})")
                    break

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"[Extended] Epoch {epoch:4d}  "
                  f"Loss={loss.item():.4f}  "
                  f"Data={parts['data']:.4f}  "
                  f"Temp={parts['temp']:.4f}  "
                  f"Spat_WD={parts['spat_wd']:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def eval_extended(model, X_torch, Y_torch, H_torch, train_idx, eval_idx,
                  alpha_window=7, alpha_ridge=1e-4):
    return eval_with_alpha_prediction(
        model, X_torch, Y_torch,
        train_idx, eval_idx,
        window_size=alpha_window,
        alpha_ridge=alpha_ridge,
        H_torch=H_torch,
    )


def train_eval_extended_config(X_torch, Y_torch, H_torch, train_idx, eval_idx,
                               neighbor_indices, masks_torch, lam,
                               lam7, mu_wd, mu_fri, mu_we, gamma,
                               lr=1e-2, num_epochs=200, device="cpu",
                               return_model=False,
                               alpha_window=7, alpha_ridge=1e-4,
                               early_stopping=False, early_stop_patience=20,
                               early_stop_min_delta=0.0, early_stop_eval_every=5):
    model = train_extended(
        X_torch, Y_torch, H_torch, train_idx,
        neighbor_indices, masks_torch,
        lam=lam, lam7=lam7, mu_wd=mu_wd, mu_fri=mu_fri, mu_we=mu_we,
        gamma=gamma,
        lr=lr, num_epochs=num_epochs, device=device, verbose=False,
        val_idx=eval_idx, early_stopping=early_stopping,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        early_stop_eval_every=early_stop_eval_every,
        alpha_window=alpha_window,
        alpha_ridge=alpha_ridge
    )
    rmse, y_true, y_pred = eval_extended(
        model, X_torch, Y_torch, H_torch, train_idx, eval_idx,
        alpha_window=alpha_window,
        alpha_ridge=alpha_ridge
    )
    result = {
        "rmse": rmse,
        "params": {
            "lam": lam,
            "lam7": lam7,
            "mu_wd": mu_wd,
            "mu_fri": mu_fri,
            "mu_we": mu_we,
            "gamma": gamma
        },
        "y_true": y_true,
        "y_pred": y_pred,
    }
    if return_model:
        result["model"] = model
    else:
        del model
    return result


def eval_best_extension_on_test(best_entry, X_torch, Y_torch, H_torch,
                                train_val_idx, test_idx,
                                neighbor_indices, masks_torch,
                                lr=1e-2, num_epochs=200, device="cpu",
                                alpha_window=7, alpha_ridge=1e-4):
    """
    Retrain the extension config on train+val and evaluate once on the test set.
    """
    if not best_entry:
        return None

    params = best_entry["params"]
    model = train_extended(
        X_torch, Y_torch, H_torch, train_val_idx,
        neighbor_indices, masks_torch,
        lam=params["lam"],
        lam7=params["lam7"],
        mu_wd=params["mu_wd"],
        mu_fri=params["mu_fri"],
        mu_we=params["mu_we"],
        gamma=params["gamma"],
        lr=lr, num_epochs=num_epochs, device=device, verbose=False
    )
    rmse, y_true, y_pred = eval_extended(
        model, X_torch, Y_torch, H_torch, train_val_idx, test_idx,
        alpha_window=alpha_window,
        alpha_ridge=alpha_ridge
    )
    return {
        "rmse": rmse,
        "params": params,
        "y_true": y_true,
        "y_pred": y_pred
    }


def evaluate_extension_impacts(X_torch, Y_torch,
                               H_zero, H_actual,
                               train_idx, val_idx,
                               neighbor_indices, masks_torch,
                               lam,
                               lam7_values,
                               mu_wd_values, mu_fri_values, mu_we_values,
                               gamma_values,
                               mu_default, lam_values=None, mu_values=None, gamma_off_value=0.0,
                               lr=1e-2, num_epochs=200, device="cpu",
                               alpha_window=7, alpha_ridge=1e-4,
                               early_stopping=False, early_stop_patience=20,
                               early_stop_min_delta=0.0, early_stop_eval_every=5):
    """
    Run isolated sweeps for each extension to quantify its standalone impact.
    Other extensions are disabled via lam7=0, mu defaults, and zeroed holidays.
    """
    results = {}
    device = torch.device(device)
    lam_candidates = lam_values if lam_values is not None else [lam]
    mu_candidates = mu_values if mu_values is not None else [mu_default]

    if lam7_values:
        history = []
        best = None
        total = len(lam_candidates) * len(mu_candidates) * len(lam7_values)
        with tqdm(total=total, desc="Weekly-only sweep") as pbar:
            for lam_candidate in lam_candidates:
                for mu_candidate in mu_candidates:
                    for lam7 in lam7_values:
                        res = train_eval_extended_config(
                            X_torch, Y_torch, H_zero, train_idx, val_idx,
                            neighbor_indices, masks_torch,
                            lam=lam_candidate,
                            lam7=lam7,
                            mu_wd=mu_candidate,
                            mu_fri=mu_candidate,
                            mu_we=mu_candidate,
                            gamma=gamma_off_value,
                            lr=lr, num_epochs=num_epochs, device=device,
                            alpha_window=alpha_window,
                            alpha_ridge=alpha_ridge,
                            early_stopping=early_stopping,
                            early_stop_patience=early_stop_patience,
                            early_stop_min_delta=early_stop_min_delta,
                            early_stop_eval_every=early_stop_eval_every
                        )
                        history.append({"lam": lam_candidate, "mu": mu_candidate, "lam7": lam7, "rmse": res["rmse"]})
                        if best is None or res["rmse"] < best["rmse"]:
                            best = res
                            pbar.set_postfix({"best_rmse": f"{res['rmse']:.4f}"})
                        pbar.update(1)
        results["weekly"] = {"best": best, "history": history}
        print(f"[Extension Sweep] Weekly-only best aRMSE={best['rmse']:.4f} "
              f"(lam={best['params'].get('lam')}, mu={best['params'].get('mu_wd')}, lam7={best['params']['lam7']})")

    if mu_wd_values or mu_fri_values or mu_we_values:
        history = []
        best = None
        if not (mu_wd_values and mu_fri_values and mu_we_values):
            print("[Extension Sweep] Day-type-only sweep skipped: "
                  "mu_wd_values, mu_fri_values, and mu_we_values must all be non-empty.")
            results["daytype"] = {"best": None, "history": []}
        else:
            combos = len(lam_candidates) * len(mu_wd_values) * len(mu_fri_values) * len(mu_we_values)
            with tqdm(total=combos, desc="Day-type-only sweep") as pbar:
                for lam_candidate in lam_candidates:
                    for mu_wd in mu_wd_values:
                        for mu_fri in mu_fri_values:
                            for mu_we in mu_we_values:
                                res = train_eval_extended_config(
                                    X_torch, Y_torch, H_zero, train_idx, val_idx,
                                    neighbor_indices, masks_torch,
                                    lam=lam_candidate,
                                    lam7=0.0,
                                    mu_wd=mu_wd,
                                    mu_fri=mu_fri,
                                    mu_we=mu_we,
                                    gamma=gamma_off_value,
                                    lr=lr, num_epochs=num_epochs, device=device,
                                    alpha_window=alpha_window,
                                    alpha_ridge=alpha_ridge,
                                    early_stopping=early_stopping,
                                    early_stop_patience=early_stop_patience,
                                    early_stop_min_delta=early_stop_min_delta,
                                    early_stop_eval_every=early_stop_eval_every
                                )
                                history.append({
                                    "lam": lam_candidate,
                                    "mu_wd": mu_wd,
                                    "mu_fri": mu_fri,
                                    "mu_we": mu_we,
                                    "rmse": res["rmse"]
                                })
                                if best is None or res["rmse"] < best["rmse"]:
                                    best = res
                                    pbar.set_postfix({"best_rmse": f"{res['rmse']:.4f}"})
                                pbar.update(1)
            results["daytype"] = {"best": best, "history": history}
            print("[Extension Sweep] Day-type-only best aRMSE="
                  f"{best['rmse']:.4f} "
                  f"(lam={best['params'].get('lam')}, "
                  f"mu_wd={best['params']['mu_wd']}, "
                  f"mu_fri={best['params']['mu_fri']}, "
                  f"mu_we={best['params']['mu_we']})")

    if gamma_values:
        history = []
        best = None
        total = len(lam_candidates) * len(mu_candidates) * len(gamma_values)
        with tqdm(total=total, desc="Holiday-only sweep") as pbar:
            for lam_candidate in lam_candidates:
                for mu_candidate in mu_candidates:
                    for gamma in gamma_values:
                        res = train_eval_extended_config(
                            X_torch, Y_torch, H_actual, train_idx, val_idx,
                            neighbor_indices, masks_torch,
                            lam=lam_candidate,
                            lam7=0.0,
                            mu_wd=mu_candidate,
                            mu_fri=mu_candidate,
                            mu_we=mu_candidate,
                            gamma=gamma,
                            lr=lr, num_epochs=num_epochs, device=device,
                            alpha_window=alpha_window,
                            alpha_ridge=alpha_ridge,
                            early_stopping=early_stopping,
                            early_stop_patience=early_stop_patience,
                            early_stop_min_delta=early_stop_min_delta,
                            early_stop_eval_every=early_stop_eval_every
                        )
                        history.append({"lam": lam_candidate, "mu": mu_candidate, "gamma": gamma, "rmse": res["rmse"]})
                        if best is None or res["rmse"] < best["rmse"]:
                            best = res
                            pbar.set_postfix({"best_rmse": f"{res['rmse']:.4f}"})
                        pbar.update(1)
        results["holiday"] = {"best": best, "history": history}
        print(f"[Extension Sweep] Holiday-only best aRMSE={best['rmse']:.4f} "
              f"(lam={best['params'].get('lam')}, mu={best['params'].get('mu_wd')}, gamma={best['params']['gamma']})")

    return results


def evaluate_extension_combinations(X_torch, Y_torch,
                                    H_zero, H_actual,
                                    train_idx, val_idx,
                                    neighbor_indices, masks_torch,
                                    lam,
                                    lam7_values, mu_wd_values, mu_fri_values, mu_we_values,
                                    gamma_values,
                                    mu_default, lam7_default, gamma_default,
                                    lr=1e-2, num_epochs=200, device="cpu",
                                    alpha_window=7, alpha_ridge=1e-4,
                                    early_stopping=False, early_stop_patience=20,
                                    early_stop_min_delta=0.0, early_stop_eval_every=5):
    """
    Evaluate pairwise extension combinations after individual tuning.
    Combinations:
      - weekly + day-type (holiday off)
      - weekly + holiday (day-type off)
      - day-type + holiday (weekly off)
    """
    results = {}
    device = torch.device(device)

    # Weekly + Day-type (holiday off)
    if lam7_values and mu_wd_values and mu_fri_values and mu_we_values:
        history = []
        best = None
        combos = len(lam7_values) * len(mu_wd_values) * len(mu_fri_values) * len(mu_we_values)
        with tqdm(total=combos, desc="Weekly+Daytype sweep") as pbar:
            for lam7 in lam7_values:
                for mu_wd in mu_wd_values:
                    for mu_fri in mu_fri_values:
                        for mu_we in mu_we_values:
                            res = train_eval_extended_config(
                                X_torch, Y_torch, H_zero, train_idx, val_idx,
                                neighbor_indices, masks_torch,
                                lam=lam,
                                lam7=lam7,
                                mu_wd=mu_wd,
                                mu_fri=mu_fri,
                                mu_we=mu_we,
                                gamma=0.0,
                                lr=lr, num_epochs=num_epochs, device=device,
                                alpha_window=alpha_window,
                                alpha_ridge=alpha_ridge,
                                early_stopping=early_stopping,
                                early_stop_patience=early_stop_patience,
                                early_stop_min_delta=early_stop_min_delta,
                                early_stop_eval_every=early_stop_eval_every
                            )
                            history.append({
                                "lam7": lam7,
                                "mu_wd": mu_wd,
                                "mu_fri": mu_fri,
                                "mu_we": mu_we,
                                "rmse": res["rmse"]
                            })
                            if best is None or res["rmse"] < best["rmse"]:
                                best = res
                                pbar.set_postfix({"best_rmse": f"{res['rmse']:.4f}"})
                            pbar.update(1)
        results["weekly_daytype"] = {"best": best, "history": history}

    # Weekly + Holiday (day-type off)
    if lam7_values and gamma_values:
        history = []
        best = None
        total = len(lam7_values) * len(gamma_values)
        with tqdm(total=total, desc="Weekly+Holiday sweep") as pbar:
            for lam7 in lam7_values:
                for gamma in gamma_values:
                    res = train_eval_extended_config(
                        X_torch, Y_torch, H_actual, train_idx, val_idx,
                        neighbor_indices, masks_torch,
                        lam=lam,
                        lam7=lam7,
                        mu_wd=mu_default,
                        mu_fri=mu_default,
                        mu_we=mu_default,
                        gamma=gamma,
                        lr=lr, num_epochs=num_epochs, device=device,
                        alpha_window=alpha_window,
                        alpha_ridge=alpha_ridge,
                        early_stopping=early_stopping,
                        early_stop_patience=early_stop_patience,
                        early_stop_min_delta=early_stop_min_delta,
                        early_stop_eval_every=early_stop_eval_every
                    )
                    history.append({"lam7": lam7, "gamma": gamma, "rmse": res["rmse"]})
                    if best is None or res["rmse"] < best["rmse"]:
                        best = res
                        pbar.set_postfix({"best_rmse": f"{res['rmse']:.4f}"})
                    pbar.update(1)
        results["weekly_holiday"] = {"best": best, "history": history}

    # Day-type + Holiday (weekly off)
    if mu_wd_values and mu_fri_values and mu_we_values and gamma_values:
        history = []
        best = None
        combos = len(mu_wd_values) * len(mu_fri_values) * len(mu_we_values) * len(gamma_values)
        with tqdm(total=combos, desc="Daytype+Holiday sweep") as pbar:
            for mu_wd in mu_wd_values:
                for mu_fri in mu_fri_values:
                    for mu_we in mu_we_values:
                        for gamma in gamma_values:
                            res = train_eval_extended_config(
                                X_torch, Y_torch, H_actual, train_idx, val_idx,
                                neighbor_indices, masks_torch,
                                lam=lam,
                                lam7=0.0,
                                mu_wd=mu_wd,
                                mu_fri=mu_fri,
                                mu_we=mu_we,
                                gamma=gamma,
                                lr=lr, num_epochs=num_epochs, device=device,
                                alpha_window=alpha_window,
                                alpha_ridge=alpha_ridge,
                                early_stopping=early_stopping,
                                early_stop_patience=early_stop_patience,
                                early_stop_min_delta=early_stop_min_delta,
                                early_stop_eval_every=early_stop_eval_every
                            )
                            history.append({
                                "mu_wd": mu_wd,
                                "mu_fri": mu_fri,
                                "mu_we": mu_we,
                                "gamma": gamma,
                                "rmse": res["rmse"]
                            })
                            if best is None or res["rmse"] < best["rmse"]:
                                best = res
                                pbar.set_postfix({"best_rmse": f"{res['rmse']:.4f}"})
                            pbar.update(1)
        results["daytype_holiday"] = {"best": best, "history": history}

    # Fill defaults for logging/consistency if any sweeps were skipped.
    results["defaults"] = {
        "lam": lam,
        "lam7": lam7_default,
        "mu_default": mu_default,
        "gamma": gamma_default
    }
    return results


# ------------------------------------------------------------------------------
# Grid Search Visualization Helpers
# ------------------------------------------------------------------------------

def visualize_baseline_grid(history, output_path="baseline_grid_search.png"):
    """
    Heatmap showing validation aRMSE as a function of lam/mu for the baseline grid search.
    """
    if not history:
        return

    df = pd.DataFrame(history)
    pivot = df.pivot(index="lam", columns="mu", values="rmse")
    pivot = pivot.sort_index().sort_index(axis=1)

    plt.figure(figsize=(6, 4))
    im = plt.imshow(
        pivot.values,
        cmap="viridis",
        origin="lower",
        aspect="auto",
        vmin=df["rmse"].min(),
        vmax=df["rmse"].max(),
    )
    plt.colorbar(im, shrink=0.8, label="aRMSE")
    plt.xticks(np.arange(len(pivot.columns)), [str(mu) for mu in pivot.columns], rotation=45)
    plt.yticks(np.arange(len(pivot.index)), [str(lam) for lam in pivot.index])
    plt.xlabel("mu")
    plt.ylabel("lam")
    plt.title("Baseline Grid Search (Validation aRMSE)")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved baseline grid-search visualization to '{output_path}'")


def visualize_extended_grid(history, output_path="extended_grid_search.png"):
    """
    Visualize the extended grid search by projecting aRMSE onto (lam7, mu_wd)
    for each gamma, after selecting the best mu_fri/mu_we combination.
    """
    if not history:
        return

    df = pd.DataFrame(history)
    reduced = (
        df.groupby(["gamma", "lam7", "mu_wd"])["rmse"]
        .min()
        .reset_index()
    )

    gammas = sorted(reduced["gamma"].unique())
    cols = min(3, len(gammas))
    rows = math.ceil(len(gammas) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)
    flat_axes = axes.ravel()

    vmin = reduced["rmse"].min()
    vmax = reduced["rmse"].max()

    im = None
    for idx, gamma in enumerate(gammas):
        ax = flat_axes[idx]
        subset = reduced[reduced["gamma"] == gamma]
        pivot = subset.pivot(index="lam7", columns="mu_wd", values="rmse")
        if pivot.empty:
            ax.set_visible(False)
            continue

        pivot = pivot.sort_index().sort_index(axis=1)
        im = ax.imshow(
            pivot.values,
            cmap="plasma",
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"gamma={gamma}")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(mu) for mu in pivot.columns], rotation=45)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(lam7) for lam7 in pivot.index])
        ax.set_xlabel("mu_wd")
    for ax in flat_axes[len(gammas):]:
        ax.axis("off")
    flat_axes[0].set_ylabel("lam7")
    if im is not None:
        fig.colorbar(im, ax=flat_axes.tolist(), shrink=0.8, label="aRMSE")
    fig.suptitle(
        "Extended Grid Search (Validation aRMSE, best mu_fri/mu_we)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved extended grid-search visualization to '{output_path}'")


def visualize_extension_impacts(baseline_rmse, impact_results,
                                output_path="extension_impacts.png",
                                title="Single-Extension Validation Comparison",
                                y_label="Validation aRMSE"):
    """
    Bar chart comparing validation aRMSE of baseline vs single-extension sweeps.
    """
    labels = ["Baseline"]
    values = [baseline_rmse]
    label_map = [
        ("weekly", "Weekly only"),
        ("daytype", "Day-type only"),
        ("holiday", "Holiday only")
    ]
    for key, label in label_map:
        if key in impact_results and impact_results[key].get("best"):
            labels.append(label)
            values.append(impact_results[key]["best"]["rmse"])

    plt.figure(figsize=(8, 4))
    bars = plt.bar(range(len(values)), values, color="#ff7f0e")
    plt.xticks(range(len(values)), labels, rotation=20)
    plt.ylabel(y_label)
    plt.title(title)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved extension comparison visualization to '{output_path}'")


def visualize_extension_combinations(baseline_rmse, extended_rmse, combo_results,
                                     output_path="extension_combos.png",
                                     title="Extension Combination Validation Comparison",
                                     y_label="Validation aRMSE"):
    """
    Bar chart comparing baseline, best single-extension sweeps, and combo sweeps.
    """
    labels = ["Baseline", "All three"]
    values = [baseline_rmse, extended_rmse]

    label_map = [
        ("weekly_daytype", "Weekly+Day-type"),
        ("weekly_holiday", "Weekly+Holiday"),
        ("daytype_holiday", "Day-type+Holiday")
    ]

    for key, label in label_map:
        entry = combo_results.get(key)
        if entry and entry.get("best"):
            labels.append(label)
            values.append(entry["best"]["rmse"])

    plt.figure(figsize=(8, 4))
    bars = plt.bar(range(len(values)), values, color="#2ca02c")
    plt.xticks(range(len(values)), labels, rotation=20)
    plt.ylabel(y_label)
    plt.title(title)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved extension combo comparison to '{output_path}'")


def visualize_baseline_sensitivity(history, output_path="baseline_parameter_sensitivity.png"):
    """
    Plot aRMSE sensitivity curves for baseline lam and mu using the grid-search history.
    """
    if not history:
        print("No baseline sensitivity data to plot.")
        return

    df = pd.DataFrame(history)
    panels = []

    if {"lam", "rmse"}.issubset(df.columns):
        agg = (
            df.dropna(subset=["lam", "rmse"])
            .groupby("lam")["rmse"]
            .min()
            .reset_index()
            .sort_values("lam")
        )
        if not agg.empty:
            panels.append({
                "title": "Temporal smoothness (lam)",
                "xlabel": "lam",
                "x": agg["lam"].to_numpy(),
                "y": agg["rmse"].to_numpy()
            })

    if {"mu", "rmse"}.issubset(df.columns):
        agg = (
            df.dropna(subset=["mu", "rmse"])
            .groupby("mu")["rmse"]
            .min()
            .reset_index()
            .sort_values("mu")
        )
        if not agg.empty:
            panels.append({
                "title": "Spatial smoothness (mu)",
                "xlabel": "mu",
                "x": agg["mu"].to_numpy(),
                "y": agg["rmse"].to_numpy()
            })

    if not panels:
        print("No baseline sensitivity data to plot.")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        ax.plot(panel["x"], panel["y"], marker="o")
        ax.set_title(panel["title"])
        ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel("Validation aRMSE")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved baseline sensitivity visualization to '{output_path}'")


def visualize_extension_parameter_sensitivity(impact_results,
                                              output_path="extension_parameter_sensitivity.png"):
    """
    Plot aRMSE sensitivity curves for each extension parameter using the sweep
    histories produced by evaluate_extension_impacts.
    """
    if not impact_results:
        print("No extension sensitivity data to plot.")
        return

    panels = []

    weekly = impact_results.get("weekly")
    if weekly and weekly.get("history"):
        df = pd.DataFrame(weekly["history"])
        if not df.empty and {"lam7", "rmse"}.issubset(df.columns):
            df = (
                df.dropna(subset=["lam7", "rmse"])
                .groupby("lam7")["rmse"]
                .min()
                .reset_index()
                .sort_values("lam7")
            )
            if not df.empty:
                panels.append({
                    "title": "Weekly coupling (lam7)",
                    "xlabel": "lam7",
                    "lines": [{
                        "x": df["lam7"].to_numpy(),
                        "y": df["rmse"].to_numpy(),
                        "label": "lam7"
                    }]
                })

    daytype = impact_results.get("daytype")
    if daytype and daytype.get("history"):
        df = pd.DataFrame(daytype["history"])
        for col, label in [("mu_wd", "Weekday mu"), ("mu_fri", "Friday mu"), ("mu_we", "Weekend mu")]:
            if col in df and "rmse" in df:
                agg = (
                    df.dropna(subset=[col, "rmse"])
                    .groupby(col)["rmse"]
                    .min()
                    .reset_index()
                    .sort_values(col)
                )
                if not agg.empty:
                    panels.append({
                        "title": f"Day-type spatial ({label})",
                        "xlabel": col,
                        "lines": [{
                            "x": agg[col].to_numpy(),
                            "y": agg["rmse"].to_numpy(),
                            "label": label
                        }]
                    })

    holiday = impact_results.get("holiday")
    if holiday and holiday.get("history"):
        df = pd.DataFrame(holiday["history"])
        if not df.empty and {"gamma", "rmse"}.issubset(df.columns):
            df = (
                df.dropna(subset=["gamma", "rmse"])
                .groupby("gamma")["rmse"]
                .min()
                .reset_index()
                .sort_values("gamma")
            )
            if not df.empty:
                panels.append({
                    "title": "Holiday penalty (gamma)",
                    "xlabel": "gamma",
                    "lines": [{
                        "x": df["gamma"].to_numpy(),
                        "y": df["rmse"].to_numpy(),
                        "label": "gamma"
                    }]
                })

    if not panels:
        print("No extension sensitivity data to plot.")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        for line in panel["lines"]:
            ax.plot(line["x"], line["y"], marker="o", label=line["label"])
        ax.set_title(panel["title"])
        ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel("Validation aRMSE")
        ax.grid(True, alpha=0.3)
        if len(panel["lines"]) > 1:
            ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved extension sensitivity visualization to '{output_path}'")


def visualize_extension_baseline_tuning(impact_results,
                                        output_path="extension_baseline_tuning.png"):
    """
    Plot how aRMSE varies with the baseline lam/mu values used during each
    extension sweep (weekly, day-type, holiday).
    """
    if not impact_results:
        print("No extension impact data; skipping baseline tuning plot.")
        return

    panels = []

    weekly = impact_results.get("weekly")
    if weekly and weekly.get("history"):
        df = pd.DataFrame(weekly["history"])
        if "lam" in df and "rmse" in df:
            agg = (
                df.dropna(subset=["lam", "rmse"])
                .groupby("lam")["rmse"]
                .min()
                .reset_index()
                .sort_values("lam")
            )
            if not agg.empty:
                panels.append({
                    "title": "Weekly extension: baseline lam",
                    "xlabel": "lam",
                    "x": agg["lam"].to_numpy(),
                    "y": agg["rmse"].to_numpy()
                })
        if "mu" in df and "rmse" in df:
            agg = (
                df.dropna(subset=["mu", "rmse"])
                .groupby("mu")["rmse"]
                .min()
                .reset_index()
                .sort_values("mu")
            )
            if not agg.empty:
                panels.append({
                    "title": "Weekly extension: baseline mu",
                    "xlabel": "mu",
                    "x": agg["mu"].to_numpy(),
                    "y": agg["rmse"].to_numpy()
                })

    daytype = impact_results.get("daytype")
    if daytype and daytype.get("history"):
        df = pd.DataFrame(daytype["history"])
        if "lam" in df and "rmse" in df:
            agg = (
                df.dropna(subset=["lam", "rmse"])
                .groupby("lam")["rmse"]
                .min()
                .reset_index()
                .sort_values("lam")
            )
            if not agg.empty:
                panels.append({
                    "title": "Day-type extension: baseline lam",
                    "xlabel": "lam",
                    "x": agg["lam"].to_numpy(),
                    "y": agg["rmse"].to_numpy()
                })

    holiday = impact_results.get("holiday")
    if holiday and holiday.get("history"):
        df = pd.DataFrame(holiday["history"])
        if "lam" in df and "rmse" in df:
            agg = (
                df.dropna(subset=["lam", "rmse"])
                .groupby("lam")["rmse"]
                .min()
                .reset_index()
                .sort_values("lam")
            )
            if not agg.empty:
                panels.append({
                    "title": "Holiday extension: baseline lam",
                    "xlabel": "lam",
                    "x": agg["lam"].to_numpy(),
                    "y": agg["rmse"].to_numpy()
                })
        if "mu" in df and "rmse" in df:
            agg = (
                df.dropna(subset=["mu", "rmse"])
                .groupby("mu")["rmse"]
                .min()
                .reset_index()
                .sort_values("mu")
            )
            if not agg.empty:
                panels.append({
                    "title": "Holiday extension: baseline mu",
                    "xlabel": "mu",
                    "x": agg["mu"].to_numpy(),
                    "y": agg["rmse"].to_numpy()
                })

    if not panels:
        print("No baseline lam/mu variation captured in extension sweeps.")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        ax.plot(panel["x"], panel["y"], marker="o")
        ax.set_title(panel["title"])
        ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel("Validation aRMSE")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved extension baseline tuning visualization to '{output_path}'")


def plot_rolling_summary(results_df, columns, labels, title, output_path):
    """
    Plot mean ± std across rolling splits for the requested columns.
    """
    if results_df.empty:
        print("No rolling results to plot.")
        return
    means = []
    stds = []
    for col in columns:
        series = results_df[col].dropna()
        means.append(series.mean() if not series.empty else np.nan)
        stds.append(series.std(ddof=0) if not series.empty else np.nan)

    x = np.arange(len(columns))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("aRMSE")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved rolling summary plot to '{output_path}'")


def aggregate_grid_history(histories, group_cols):
    """
    Aggregate grid-search histories across splits by averaging aRMSE per config.
    """
    frames = []
    for hist in histories:
        if not hist:
            continue
        frames.append(pd.DataFrame(hist))
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    if "rmse" not in df.columns:
        return []
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        return []
    agg = df.groupby(group_cols)["rmse"].mean().reset_index()
    return agg.to_dict("records")


def _aggregate_param_min_over_splits(histories, param_col):
    """
    For each split, take min RMSE per param value, then average across splits.
    """
    per_split = []
    for hist in histories:
        if not hist:
            continue
        df = pd.DataFrame(hist)
        if param_col not in df.columns or "rmse" not in df.columns:
            continue
        df = df.dropna(subset=[param_col, "rmse"])
        if df.empty:
            continue
        per_split.append(df.groupby(param_col)["rmse"].min())
    if not per_split:
        return []
    combined = pd.concat(per_split, axis=1)
    mean = combined.mean(axis=1)
    return [{param_col: idx, "rmse": float(val)} for idx, val in mean.items()]


def aggregate_extension_impacts_histories(impacts_list):
    """
    Aggregate extension impact histories into a single history per extension.
    """
    weekly_histories = []
    daytype_histories = []
    holiday_histories = []
    for impacts in impacts_list:
        weekly_histories.append(impacts.get("weekly", {}).get("history", []))
        daytype_histories.append(impacts.get("daytype", {}).get("history", []))
        holiday_histories.append(impacts.get("holiday", {}).get("history", []))

    weekly_history = []
    weekly_history += _aggregate_param_min_over_splits(weekly_histories, "lam7")
    weekly_history += _aggregate_param_min_over_splits(weekly_histories, "lam")
    weekly_history += _aggregate_param_min_over_splits(weekly_histories, "mu")

    daytype_history = []
    daytype_history += _aggregate_param_min_over_splits(daytype_histories, "mu_wd")
    daytype_history += _aggregate_param_min_over_splits(daytype_histories, "mu_fri")
    daytype_history += _aggregate_param_min_over_splits(daytype_histories, "mu_we")
    daytype_history += _aggregate_param_min_over_splits(daytype_histories, "lam")

    holiday_history = []
    holiday_history += _aggregate_param_min_over_splits(holiday_histories, "gamma")
    holiday_history += _aggregate_param_min_over_splits(holiday_histories, "lam")
    holiday_history += _aggregate_param_min_over_splits(holiday_histories, "mu")

    return {
        "weekly": {"history": weekly_history},
        "daytype": {"history": daytype_history},
        "holiday": {"history": holiday_history},
    }

# ------------------------------------------------------------------------------
# Paper-Style Pattern Diagnostics (Figures 8–9)
# ------------------------------------------------------------------------------

def plot_weekly_periodicity(Y, max_lag=100, output_path="paper_weekly_periodicity.png"):
    """
    Fig. 8(a)-style diagnostic: average |c_t - c_{t+Δt}| vs Δt (days),
    where Y is daily crime counts per region.

    Y: numpy array of shape (K, N)
    """
    Y = np.asarray(Y)
    if Y.ndim != 2:
        raise ValueError(f"Expected Y to have shape (K, N); got {Y.shape}")

    K, _ = Y.shape
    max_lag = int(max_lag)
    if K < 2:
        raise ValueError("Need at least 2 time steps to compute lag differences.")
    max_lag = max(1, min(max_lag, K - 1))

    lags = np.arange(1, max_lag + 1)
    avg_abs_diff = np.empty_like(lags, dtype=float)

    for idx, lag in enumerate(lags):
        diffs = np.abs(Y[lag:] - Y[:-lag])
        avg_abs_diff[idx] = float(np.nanmean(diffs))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lags, avg_abs_diff, color="teal", linewidth=1.5)
    for m in range(7, max_lag + 1, 7):
        ax.axvline(m, color="black", alpha=0.08, linewidth=1)
    ax.set_title("Weekly Periodicity (Avg |c(t) − c(t+Δt)|)")
    ax.set_xlabel("Δt (days)")
    ax.set_ylabel("Avg absolute difference")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved weekly periodicity plot to '{output_path}'")


def plot_day_of_year(dates, Y, output_path="paper_day_of_year.png"):
    """
    Fig. 8(b)-style diagnostic: average citywide daily crime vs day-of-year.

    dates: pandas Index/DatetimeIndex of length K
    Y: numpy array of shape (K, N)
    """
    dates = pd.to_datetime(pd.Index(dates))
    Y = np.asarray(Y)
    if Y.ndim != 2 or len(dates) != Y.shape[0]:
        raise ValueError(f"Expected dates length K and Y shape (K, N); got len(dates)={len(dates)}, Y={Y.shape}")

    citywide_daily = np.nansum(Y, axis=1)
    series = pd.Series(citywide_daily, index=dates).sort_index()

    doy = series.index.dayofyear
    doy_avg = series.groupby(doy).mean()

    x = np.arange(1, 367)
    y = np.full_like(x, np.nan, dtype=float)
    present = doy_avg.index.values.astype(int)
    y[present - 1] = doy_avg.values

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, color="green", linewidth=1.5)
    ax.set_title("Day-of-Year Pattern (Avg Citywide Daily Crime)")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Avg daily crime (sum over regions)")
    ax.grid(True, alpha=0.2)

    # Month ticks (use a non-leap baseline year for readability)
    baseline = pd.date_range("2019-01-01", "2019-12-31", freq="MS")
    month_starts = baseline.dayofyear.values
    ax.set_xticks(month_starts)
    ax.set_xticklabels([d.strftime("%b") for d in baseline])

    # Light reference lines for fixed-date holidays mentioned in the paper
    new_year = pd.Timestamp("2019-01-01").dayofyear
    christmas = pd.Timestamp("2019-12-25").dayofyear
    ax.axvline(new_year, color="black", alpha=0.08, linewidth=1)
    ax.axvline(christmas, color="black", alpha=0.08, linewidth=1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved day-of-year plot to '{output_path}'")


def plot_dayofweek_spatial(grid_active, regions, dates, Y, output_path="paper_dayofweek_spatial.png"):
    """
    Fig. 9-style diagnostic: 7 maps (Mon..Sun) showing average crime per region.

    grid_active: GeoDataFrame with 'region_id' and 'geometry'
    regions: region_ids aligned with Y's 2nd dimension
    dates: pandas Index/DatetimeIndex of length K
    Y: numpy array of shape (K, N)
    """
    dates = pd.to_datetime(pd.Index(dates))
    Y = np.asarray(Y)
    if Y.ndim != 2 or len(dates) != Y.shape[0]:
        raise ValueError(f"Expected dates length K and Y shape (K, N); got len(dates)={len(dates)}, Y={Y.shape}")
    if "region_id" not in grid_active.columns:
        raise ValueError("grid_active must contain a 'region_id' column for joining to regions.")

    regions = np.asarray(regions)
    grid_plot = grid_active.set_index("region_id").loc[regions].reset_index()

    dow = dates.dayofweek.values  # 0=Mon ... 6=Sun
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    region_means = []
    for d in range(7):
        mask = dow == d
        if not mask.any():
            region_means.append(np.full(Y.shape[1], np.nan))
        else:
            region_means.append(np.nanmean(Y[mask], axis=0))

    finite = [v[np.isfinite(v)] for v in region_means if np.isfinite(v).any()]
    all_vals = np.concatenate(finite, axis=0) if finite else np.array([], dtype=float)
    vmax = float(np.nanpercentile(all_vals, 99)) if all_vals.size else 1.0
    vmax = max(vmax, 1.0)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for d in range(7):
        ax = axes[d]
        plot_df = grid_plot.copy()
        plot_df["avg_crime"] = region_means[d]
        plot_df.plot(
            column="avg_crime",
            ax=ax,
            legend=(d == 6),
            cmap="Blues",
            vmin=0.0,
            vmax=vmax,
            missing_kwds={"color": "lightgrey", "label": "No data"},
        )
        ax.set_title(dow_names[d])
        ax.axis("off")

    axes[7].axis("off")
    plt.suptitle("Spatial Distribution by Day of Week (Avg Crime per Region)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved day-of-week spatial plot to '{output_path}'")


def summarize_daytype_spatial_differences(regions, dates, Y,
                                          output_csv="paper_daytype_spatial_summary.csv",
                                          top_n=10):
    """
    Quantify spatial differences between day types with region-wise averages.

    Outputs a CSV with per-region means and pairwise differences, and prints
    aggregate summary statistics that can be reported numerically.
    """
    dates = pd.to_datetime(pd.Index(dates))
    Y = np.asarray(Y)
    if Y.ndim != 2 or len(dates) != Y.shape[0]:
        raise ValueError(f"Expected dates length K and Y shape (K, N); got len(dates)={len(dates)}, Y={Y.shape}")

    dow = dates.dayofweek.values  # 0=Mon ... 6=Sun
    masks = {
        "wd": (dow >= 0) & (dow <= 3),   # Mon-Thu
        "fri": (dow == 4),
        "we": (dow >= 5)                 # Sat-Sun
    }

    regions = np.asarray(regions)
    stats = {"region_id": regions}
    for key, mask in masks.items():
        if mask.any():
            stats[f"mean_{key}"] = np.nanmean(Y[mask], axis=0)
        else:
            stats[f"mean_{key}"] = np.full(Y.shape[1], np.nan)

    df = pd.DataFrame(stats)
    df["diff_fri_wd"] = df["mean_fri"] - df["mean_wd"]
    df["diff_we_wd"] = df["mean_we"] - df["mean_wd"]
    df["diff_we_fri"] = df["mean_we"] - df["mean_fri"]
    df["absdiff_fri_wd"] = df["diff_fri_wd"].abs()
    df["absdiff_we_wd"] = df["diff_we_wd"].abs()
    df["absdiff_we_fri"] = df["diff_we_fri"].abs()

    df.to_csv(output_csv, index=False)

    def _series_stats(series):
        series = series[np.isfinite(series)]
        if series.empty:
            return {"mean": np.nan, "median": np.nan, "std": np.nan, "pct_positive": np.nan, "pct_negative": np.nan}
        return {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=0)),
            "pct_positive": float((series > 0).mean() * 100.0),
            "pct_negative": float((series < 0).mean() * 100.0),
        }

    def _corr(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            return np.nan
        return float(np.corrcoef(a[mask], b[mask])[0, 1])

    summary = {
        "fri_vs_wd": _series_stats(df["diff_fri_wd"]),
        "we_vs_wd": _series_stats(df["diff_we_wd"]),
        "we_vs_fri": _series_stats(df["diff_we_fri"]),
        "corr_fri_wd": _corr(df["mean_fri"].values, df["mean_wd"].values),
        "corr_we_wd": _corr(df["mean_we"].values, df["mean_wd"].values),
        "corr_we_fri": _corr(df["mean_we"].values, df["mean_fri"].values),
    }

    print("\n--- Day-Type Spatial Difference Summary ---")
    print(f"Saved per-region stats to '{output_csv}'")
    for key, stats in summary.items():
        if key.startswith("corr_"):
            print(f"{key}: {stats:.3f}" if np.isfinite(stats) else f"{key}: n/a")
            continue
        print(
            f"{key}: mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
            f"std={stats['std']:.3f}, +%={stats['pct_positive']:.1f}, -%={stats['pct_negative']:.1f}"
        )

    if top_n and top_n > 0:
        top_cols = ["absdiff_fri_wd", "absdiff_we_wd", "absdiff_we_fri"]
        for col in top_cols:
            top = df.nlargest(top_n, col)[["region_id", col]]
            print(f"Top {top_n} regions by {col}:")
            print(top.to_string(index=False))

    return df, summary


def generate_paper_pattern_plots(Y, dates, grid_active, regions, max_lag=100, output_prefix="paper"):
    """
    Generate all three paper-style pattern plots (Fig. 8a, 8b, Fig. 9).
    """
    plot_weekly_periodicity(Y, max_lag=max_lag, output_path=f"{output_prefix}_weekly_periodicity.png")
    plot_day_of_year(dates, Y, output_path=f"{output_prefix}_day_of_year.png")
    plot_dayofweek_spatial(grid_active, regions, dates, Y, output_path=f"{output_prefix}_dayofweek_spatial.png")

# ------------------------------------------------------------------------------
# Main Execution Block
# ------------------------------------------------------------------------------

if __name__ == "__main__":

    # 1. Import feature matrix and grid
    print("Loading data...")
    X_df = pd.read_csv('data/FEATURE_MATRIX.csv')
    grid_active = gpd.read_file("data/nyc_grid_2km_active.shp")

    feature_cols = ["sas_count", "311_count", "checkin_count",
                    "taxi_count", "PRCP", "SNOW", "TMIN", "TMAX"]

    # 2. Reshape data and get masks
    X_tensor, Y, regions, dates, masks_np = reshape_for_tcp(
        X_df,
        feature_cols=feature_cols,
        target_col="complaint_count",
    )

    # 3. Setup PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_torch = torch.from_numpy(X_tensor).float().to(device)
    Y_torch = torch.from_numpy(Y).float().to(device)

    # New: Create Holiday Tensor
    H_torch = get_holiday_tensor(dates, device)
    H_torch_zero = torch.zeros_like(H_torch)

    K, N, M = X_torch.shape

    # ------------------------------------------------------------------------------
    # Paper-style temporal/spatial pattern diagnostics (Figures 8–9)
    # ------------------------------------------------------------------------------
    print("\n--- Generating Paper-Style Pattern Plots ---")
    generate_paper_pattern_plots(
        Y=Y,
        dates=dates,
        grid_active=grid_active,
        regions=regions,
        max_lag=100,
        output_prefix="paper",
    )
    summarize_daytype_spatial_differences(
        regions=regions,
        dates=dates,
        Y=Y,
        output_csv="paper_daytype_spatial_summary.csv",
        top_n=10,
    )

    # ------------------------------------------------------------------------------
    # Rolling (walk-forward) CV (paper setup)
    # ------------------------------------------------------------------------------
    rolling_train_size = max(1, int(0.7 * K))
    rolling_val_size = max(1, int(0.15 * K))
    rolling_test_size = max(1, K - rolling_train_size - rolling_val_size)
    rolling_step_size = rolling_val_size

    # Early-stopping config (used during hyperparameter selection)
    use_early_stopping = True
    early_stop_patience = 20
    early_stop_min_delta = 0.0
    early_stop_eval_every = 5

    # 4. Build Neighbor Indices
    neighbor_pairs = build_neighbor_pairs(grid_active, regions)
    neighbor_indices = process_neighbor_indices(neighbor_pairs, device=device)

    # Sliding-window / alpha-stage configuration (paper uses alpha forecasting)
    alpha_window = 28
    alpha_ridge = 1e-3

    # Training budget controls:
    # - sweeps: quick/diagnostic runs to compare settings
    # - final: full training for selected configs
    sweep_num_epochs = 100
    final_num_epochs = 200

    # Baseline model (no holidays, no day-type spatial structure) + grid search
    lam_grid = [1000000, 2000000, 3000000, 5000000]
    mu_grid = [0, 1000, 2000, 3000]

    # Extended model grids
    lam7_grid_ext = [0, 5000, 10000]
    mu_wd_values = [0, 200, 500]
    mu_fri_values = [0, 1000, 2000]
    mu_we_values = [0, 1000, 5000]
    gamma_grid = [0, 1, 5]

    masks_torch = {k: torch.from_numpy(v).to(device) for k, v in masks_np.items()}

    splits = build_rolling_splits(
        K,
        train_size=rolling_train_size,
        val_size=rolling_val_size,
        test_size=rolling_test_size,
        step_size=rolling_step_size,
        device=device,
    )
    if not splits:
        raise ValueError("Rolling split configuration produced no splits.")

    rolling_metrics = []
    last_split_baseline = None
    last_split_test_idx = None
    baseline_histories = []
    extension_impacts_list = []
    extended_grid_histories = []

    for split_idx, split in enumerate(splits, start=1):
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        test_idx = split["test_idx"]
        train_val_idx = split["train_val_idx"]

        print(f"\n--- Rolling Split {split_idx}/{len(splits)} ---")

        baseline_search = grid_search_baseline(
            X_torch, Y_torch, train_idx, val_idx, neighbor_indices,
            lam_values=lam_grid, mu_values=mu_grid,
            lr=1e-2, num_epochs=sweep_num_epochs, device=device,
            viz_path=None,
            alpha_window=alpha_window,
            alpha_ridge=alpha_ridge,
            early_stopping=use_early_stopping,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            early_stop_eval_every=early_stop_eval_every
        )
        baseline_histories.append(baseline_search.get("history", []))

        best_baseline_params = baseline_search["params"]
        baseline_model = train_baseline(
            X_torch, Y_torch, train_val_idx, neighbor_indices,
            lam=best_baseline_params["lam"],
            mu=best_baseline_params["mu"],
            lr=1e-2, num_epochs=final_num_epochs, device=device, verbose=False
        )
        rmse_base_test, y_true_base_test, y_pred_base_test = eval_baseline(
            baseline_model, X_torch, Y_torch, train_val_idx, test_idx,
            alpha_window=alpha_window,
            alpha_ridge=alpha_ridge
        )
        if split_idx == len(splits):
            last_split_baseline = (y_true_base_test, y_pred_base_test)
            last_split_test_idx = test_idx

        lam_best = best_baseline_params["lam"]
        mu_best = best_baseline_params["mu"]

        extension_impacts = evaluate_extension_impacts(
            X_torch, Y_torch,
            H_torch_zero, H_torch,
            train_idx, val_idx,
            neighbor_indices, masks_torch,
            lam=lam_best,
            lam_values=lam_grid,
            lam7_values=lam7_grid_ext,
            mu_wd_values=mu_wd_values,
            mu_fri_values=mu_fri_values,
            mu_we_values=mu_we_values,
            gamma_values=gamma_grid,
            mu_default=mu_best,
            mu_values=mu_grid,
            gamma_off_value=0.0,
            lr=1e-2, num_epochs=sweep_num_epochs, device=device,
            alpha_window=alpha_window,
            alpha_ridge=alpha_ridge,
            early_stopping=use_early_stopping,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            early_stop_eval_every=early_stop_eval_every
        )
        extension_impacts_list.append(extension_impacts)

        extension_impacts_test = {}
        for key, H_eval in [
            ("weekly", H_torch_zero),
            ("daytype", H_torch_zero),
            ("holiday", H_torch),
        ]:
            best_entry = extension_impacts.get(key, {}).get("best")
            test_entry = eval_best_extension_on_test(
                best_entry,
                X_torch, Y_torch, H_eval,
                train_val_idx, test_idx,
                neighbor_indices, masks_torch,
                lr=1e-2, num_epochs=final_num_epochs, device=device,
                alpha_window=alpha_window,
                alpha_ridge=alpha_ridge
            )
            if test_entry:
                extension_impacts_test[key] = test_entry

        combo_results = evaluate_extension_combinations(
            X_torch, Y_torch,
            H_torch_zero, H_torch,
            train_idx, val_idx,
            neighbor_indices, masks_torch,
            lam=lam_best,
            lam7_values=lam7_grid_ext,
            mu_wd_values=mu_wd_values,
            mu_fri_values=mu_fri_values,
            mu_we_values=mu_we_values,
            gamma_values=gamma_grid,
            mu_default=mu_best,
            lam7_default=0.0,
            gamma_default=0.0,
            lr=1e-2, num_epochs=sweep_num_epochs, device=device,
            alpha_window=alpha_window,
            alpha_ridge=alpha_ridge,
            early_stopping=use_early_stopping,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            early_stop_eval_every=early_stop_eval_every
        )

        combo_results_test = {}
        for key, H_eval in [
            ("weekly_daytype", H_torch_zero),
            ("weekly_holiday", H_torch),
            ("daytype_holiday", H_torch),
        ]:
            best_entry = combo_results.get(key, {}).get("best")
            test_entry = eval_best_extension_on_test(
                best_entry,
                X_torch, Y_torch, H_eval,
                train_val_idx, test_idx,
                neighbor_indices, masks_torch,
                lr=1e-2, num_epochs=final_num_epochs, device=device,
                alpha_window=alpha_window,
                alpha_ridge=alpha_ridge
            )
            if test_entry:
                combo_results_test[key] = test_entry

        extended_search = grid_search_extended(
            X_torch, Y_torch, H_torch, train_idx, val_idx,
            neighbor_indices, masks_torch,
            lam=lam_best,
            lam_values=lam_grid,
            lam7_values=lam7_grid_ext,
            mu_wd_values=mu_wd_values,
            mu_fri_values=mu_fri_values,
            mu_we_values=mu_we_values,
            gamma_values=gamma_grid,
            lr=1e-2, num_epochs=sweep_num_epochs, device=device,
            viz_path=None,
            alpha_window=alpha_window,
            alpha_ridge=alpha_ridge,
            early_stopping=use_early_stopping,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            early_stop_eval_every=early_stop_eval_every
        )
        extended_grid_histories.append(extended_search.get("history", []))

        best_ext_params = extended_search["params"]
        extended_model = train_extended(
            X_torch, Y_torch, H_torch, train_val_idx,
            neighbor_indices, masks_torch,
            lam=best_ext_params["lam"],
            lam7=best_ext_params["lam7"],
            mu_wd=best_ext_params["mu_wd"],
            mu_fri=best_ext_params["mu_fri"],
            mu_we=best_ext_params["mu_we"],
            gamma=best_ext_params["gamma"],
            lr=1e-2, num_epochs=final_num_epochs, device=device, verbose=False
        )
        rmse_ext_test, _, _ = eval_extended(
            extended_model, X_torch, Y_torch, H_torch, train_val_idx, test_idx,
            alpha_window=alpha_window,
            alpha_ridge=alpha_ridge
        )

        split_metrics = {
            "split": split_idx,
            "baseline_val": baseline_search.get("rmse"),
            "baseline": rmse_base_test,
            "weekly_val": extension_impacts.get("weekly", {}).get("best", {}).get("rmse"),
            "daytype_val": extension_impacts.get("daytype", {}).get("best", {}).get("rmse"),
            "holiday_val": extension_impacts.get("holiday", {}).get("best", {}).get("rmse"),
            "weekly": extension_impacts_test.get("weekly", {}).get("rmse"),
            "daytype": extension_impacts_test.get("daytype", {}).get("rmse"),
            "holiday": extension_impacts_test.get("holiday", {}).get("rmse"),
            "weekly_daytype_val": combo_results.get("weekly_daytype", {}).get("best", {}).get("rmse"),
            "weekly_holiday_val": combo_results.get("weekly_holiday", {}).get("best", {}).get("rmse"),
            "daytype_holiday_val": combo_results.get("daytype_holiday", {}).get("best", {}).get("rmse"),
            "weekly_daytype": combo_results_test.get("weekly_daytype", {}).get("rmse"),
            "weekly_holiday": combo_results_test.get("weekly_holiday", {}).get("rmse"),
            "daytype_holiday": combo_results_test.get("daytype_holiday", {}).get("rmse"),
            "all_three_val": extended_search.get("rmse"),
            "all_three": rmse_ext_test,
        }
        rolling_metrics.append(split_metrics)

    results_df = pd.DataFrame(rolling_metrics)
    results_df.to_csv("rolling_cv_summary.csv", index=False)
    print("\n--- Rolling CV Summary ---")
    metric_cols = [c for c in results_df.columns if c not in {"split"}]
    for col in metric_cols:
        series = results_df[col].dropna()
        if series.empty:
            continue
        print(f"{col}: mean={series.mean():.4f}, std={series.std(ddof=0):.4f}")

    plot_rolling_summary(
        results_df,
        columns=["baseline", "weekly", "daytype", "holiday", "all_three"],
        labels=["Baseline", "Weekly", "Day-type", "Holiday", "All three"],
        title="Rolling CV Test Comparison (Mean ± Std)",
        output_path="result/rolling_test_comparison.png",
    )

    plot_rolling_summary(
        results_df,
        columns=["baseline", "weekly_daytype", "weekly_holiday", "daytype_holiday", "all_three"],
        labels=["Baseline", "Weekly+Daytype", "Weekly+Holiday", "Daytype+Holiday", "All three"],
        title="Rolling CV Test Combo Comparison (Mean ± Std)",
        output_path="result/rolling_test_combos.png",
    )

    plot_rolling_summary(
        results_df,
        columns=["baseline_val", "weekly_val", "daytype_val", "holiday_val", "all_three_val"],
        labels=["Baseline", "Weekly", "Day-type", "Holiday", "All three"],
        title="Rolling CV Validation Tuning (Mean ± Std)",
        output_path="result/rolling_val_comparison.png",
    )

    agg_baseline_history = aggregate_grid_history(
        baseline_histories, ["lam", "mu"]
    )
    if agg_baseline_history:
        visualize_baseline_grid(agg_baseline_history, "result/rolling_baseline_grid.png")
        visualize_baseline_sensitivity(
            agg_baseline_history,
            output_path="result/rolling_baseline_sensitivity.png"
        )

    agg_extended_history = aggregate_grid_history(
        extended_grid_histories,
        ["lam", "lam7", "mu_wd", "mu_fri", "mu_we", "gamma"]
    )
    if agg_extended_history:
        visualize_extended_grid(agg_extended_history, "result/rolling_extended_grid.png")

    agg_impacts = aggregate_extension_impacts_histories(extension_impacts_list)
    if agg_impacts:
        visualize_extension_parameter_sensitivity(
            agg_impacts,
            output_path="result/rolling_extension_parameter_sensitivity.png"
        )
        visualize_extension_baseline_tuning(
            agg_impacts,
            output_path="result/rolling_extension_baseline_tuning.png"
        )

    # Paper qualitative figure: baseline spatial distribution for the first test day
    # of the last rolling split.
    if last_split_baseline is not None and last_split_test_idx is not None:
        y_true_last, y_pred_last = last_split_baseline
        K_test = len(last_split_test_idx)
        Y_true_reshaped = y_true_last.reshape(K_test, N)
        Y_pred_reshaped = y_pred_last.reshape(K_test, N)

        day_idx_to_plot = 0
        plot_global_idx = last_split_test_idx[day_idx_to_plot].item()
        date_str = str(dates[plot_global_idx].date())

        plot_df = grid_active.copy()
        plot_df["Actual"] = Y_true_reshaped[day_idx_to_plot]
        plot_df["Predicted"] = Y_pred_reshaped[day_idx_to_plot]

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        plot_df.plot(column="Actual", ax=axes[0], legend=True, cmap="OrRd", vmin=0, vmax=20)
        axes[0].set_title(f"Baseline Actual - {date_str}")
        axes[0].axis("off")

        plot_df.plot(column="Predicted", ax=axes[1], legend=True, cmap="OrRd", vmin=0, vmax=20)
        axes[1].set_title(f"Baseline Predicted - {date_str}")
        axes[1].axis("off")

        plt.suptitle("Baseline Spatial Distribution (Last Split, First Test Day)", fontsize=16)
        plt.tight_layout()
        plt.savefig("visualization_heatmap_baseline_last_split.png")
        print("Saved heatmap to 'visualization_heatmap_baseline_last_split.png'")
