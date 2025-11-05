# ============================================================
# 📦 plot_utils.py — Helper functions for Decision Tree plots
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------
# 🌿 CLASSIFICATION PLOTS
# ------------------------------------------------------------
def plot_decision_boundaries(df, y, label_names=None, top_features=None,
                             max_points=1000, mesh_step=0.5, **params):
    """
    Efficiently plot up to 6 decision boundary graphs for classification problems.

    Handles categorical/numeric targets and features.
    Automatically subsamples large datasets, and fixes legend mismatches.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.
    y : array-like
        Target variable (categorical or numeric classification).
    label_names : list, optional
        Class names for legend. Auto-inferred if None.
    top_features : list, optional
        Features to plot. If None, uses all valid features.
    max_points : int, default=2000
        Max data points for plotting (subsample if too large).
    mesh_step : float, default=0.2
        Mesh grid step size (higher = faster, coarser).
    **params : dict
        Extra DecisionTreeClassifier parameters.
    """
    figs = []

    # --- Encode target (categorical or numeric classification) ---
    y = pd.Series(y).reset_index(drop=True)

    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype("category")
        class_names = list(y.cat.categories)
        y_encoded = y.cat.codes.values
    else:
        unique_vals = np.unique(y)
        class_names = [str(v) for v in unique_vals]
        y_encoded = pd.Categorical(y, categories=unique_vals).codes

    # Use provided label_names if valid, else inferred ones
    if label_names is None or len(label_names) != len(class_names):
        label_names = class_names

    # --- Encode features (categorical to numeric) ---
    X = df.copy().reset_index(drop=True)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes

    # --- Clean NaNs and infinities ---
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    if X.empty:
        print("⚠️ No valid numeric data found after cleaning. Cannot plot decision boundaries.")
        return []

    # --- Subsample for speed ---
    if len(X) > max_points:
        sample_idx = np.random.choice(len(X), max_points, replace=False)
        X = X.iloc[sample_idx].reset_index(drop=True)
        y_encoded = y_encoded[sample_idx]

    # --- Validate feature list ---
    if top_features is None:
        top_features = X.columns.tolist()
    top_features = [f for f in top_features if f in X.columns]

    if len(top_features) < 2:
        print("⚠️ Need at least two valid features to plot decision boundaries.")
        return []

    # --- Build feature pairs (up to 6) ---
    pairs = list(combinations(top_features, 2))[:6]

    # --- Plot each pair ---
    for (f1, f2) in pairs:
        X_pair = X[[f1, f2]].values

        if np.unique(X_pair[:, 0]).size < 2 or np.unique(X_pair[:, 1]).size < 2:
            continue

        clf = DecisionTreeClassifier(**params)
        clf.fit(X_pair, y_encoded)

        # Mesh grid using data percentiles for compact range
        x_min, x_max = np.percentile(X_pair[:, 0], [1, 99])
        y_min, y_max = np.percentile(X_pair[:, 1], [1, 99])
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, mesh_step),
            np.arange(y_min, y_max, mesh_step)
        )

        # Predict over grid
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # Define color maps
        n_classes = len(np.unique(y_encoded))
        cmap_light = ListedColormap(plt.cm.tab10.colors[:n_classes])
        cmap_bold = plt.cm.tab10.colors[:n_classes]

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.contourf(xx, yy, Z.astype(float), alpha=0.4, cmap=cmap_light)

        # Scatter actual points by class with proper names
        for class_index, color in enumerate(cmap_bold[:n_classes]):
            mask = y_encoded == class_index
            if np.any(mask):
                ax.scatter(X_pair[mask, 0], X_pair[mask, 1],
                           c=[color], edgecolor="black", s=30,
                           label=label_names[class_index])

        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.set_title(f"Decision Boundary: {f1} vs {f2}")
        ax.legend(title="Classes")
        figs.append(fig)

    if not figs:
        print("⚠️ No valid feature pairs found for plotting. Try using different features.")
    return figs
# ------------------------------------------------------------
# 📈 REGRESSION PLOTS (2D decision surface style)
# ------------------------------------------------------------
def plot_regression_surfaces(df, y_reg, top_features,
                             max_points=1000, mesh_step=0.5, **params):
    """
    Plot up to 6 regression decision boundary-like curves.

    This mimics decision boundaries for regression tasks by plotting
    contour levels of predicted values from a DecisionTreeRegressor.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe.
    y_reg : array-like
        Continuous target variable.
    top_features : list
        Features to use for 2D plots.
    max_points : int, default=1000
        Maximum number of samples to use for speed.
    mesh_step : float, default=0.5
        Step size for the mesh grid.
    **params : dict
        Extra parameters for DecisionTreeRegressor.
    """
    figs = []

    # --- Clean and subsample ---
    X = df.copy().reset_index(drop=True)
    y_reg = pd.Series(y_reg).reset_index(drop=True)

    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes

    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    if len(X) > max_points:
        idx = np.random.choice(len(X), max_points, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y_reg = y_reg.iloc[idx].reset_index(drop=True)

    # --- Validate features ---
    top_features = [f for f in top_features if f in X.columns]
    if len(top_features) < 2:
        print("⚠️ Need at least two valid features to plot regression decision curves.")
        return []

    pairs = list(combinations(top_features, 2))[:6]

    # --- Plot for each pair ---
    for (f1, f2) in pairs:
        X_pair = X[[f1, f2]].values
        if np.unique(X_pair[:, 0]).size < 2 or np.unique(X_pair[:, 1]).size < 2:
            continue

        reg = DecisionTreeRegressor(**params)
        reg.fit(X_pair, y_reg)

        # Compact mesh grid
        x_min, x_max = np.percentile(X_pair[:, 0], [1, 99])
        y_min, y_max = np.percentile(X_pair[:, 1], [1, 99])
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, mesh_step),
            np.arange(y_min, y_max, mesh_step)
        )
        Z = reg.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # --- Plot like decision boundaries ---
        fig, ax = plt.subplots(figsize=(7, 5))
        contour = ax.contour(xx, yy, Z, levels=10, cmap='viridis', linewidths=1.5)
        ax.clabel(contour, inline=True, fontsize=8)

        scatter = ax.scatter(X_pair[:, 0], X_pair[:, 1],
                             c=y_reg, cmap='viridis', edgecolor="k", s=30)

        fig.colorbar(scatter, ax=ax, label="Actual Value")

        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.set_title(f"Regression Decision Curve: {f1} vs {f2}")
        figs.append(fig)

    if not figs:
        print("⚠️ No valid feature pairs found for plotting regression curves.")
    return figs

