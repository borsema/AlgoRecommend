import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from matplotlib.colors import ListedColormap
import streamlit as st

def plot_knn_boundaries(X, y, top_features, params, class_names=None, max_points=1000):
    X, y = X.copy(), pd.Series(y).reset_index(drop=True)

    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype("category")
        class_names = class_names or list(y.cat.categories)
        y_encoded = y.cat.codes
    else:
        unique_vals = np.unique(y)
        class_names = class_names or [str(v) for v in unique_vals]
        y_encoded = pd.Categorical(y, categories=unique_vals).codes

    if len(X) > max_points:
        idx = np.random.choice(len(X), max_points, replace=False)
        X = X.iloc[idx]
        y_encoded = y_encoded[idx]

    top_features = [f for f in top_features if f in X.columns]
    if len(top_features) < 2:
        return []

    figs = []
    for f1, f2 in list(combinations(top_features, 2))[:6]:
        X_pair = X[[f1, f2]].values
        if np.unique(X_pair[:, 0]).size < 2 or np.unique(X_pair[:, 1]).size < 2:
            continue

        clf = KNeighborsClassifier(**params)
        clf.fit(X_pair, y_encoded)

        x_min, x_max = X_pair[:, 0].min(), X_pair[:, 0].max()
        y_min, y_max = X_pair[:, 1].min(), X_pair[:, 1].max()
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        n_classes = len(np.unique(y_encoded))
        cmap_light = ListedColormap(plt.cm.tab10.colors[:n_classes])
        cmap_bold = plt.cm.tab10.colors[:n_classes]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
        for idx, color in enumerate(cmap_bold[:n_classes]):
            mask = y_encoded == idx
            ax.scatter(X_pair[mask, 0], X_pair[mask, 1], c=[color], edgecolor="black", s=30, label=class_names[idx])
        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.set_title(f"KNN Boundary: {f1} vs {f2}")
        ax.legend(title="Classes")
        figs.append(fig)

    return figs

def plot_knn_surfaces(X, y_reg, top_features, params, max_points=1000):
    X = X.copy().reset_index(drop=True)
    y_reg = pd.Series(y_reg).reset_index(drop=True)

    if len(X) > max_points:
        idx = np.random.choice(len(X), max_points, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y_reg = y_reg.iloc[idx].reset_index(drop=True)

    top_features = [f for f in top_features if f in X.columns]
    if len(top_features) < 2:
        return []

    figs = []
    for f1, f2 in list(combinations(top_features, 2))[:6]:
        X_pair = X[[f1, f2]].values
        if np.unique(X_pair[:, 0]).size < 2 or np.unique(X_pair[:, 1]).size < 2:
            continue

        reg = KNeighborsRegressor(**params)
        reg.fit(X_pair, y_reg)

        x_min, x_max = X_pair[:, 0].min(), X_pair[:, 0].max()
        y_min, y_max = X_pair[:, 1].min(), X_pair[:, 1].max()
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )

        Z = reg.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(7, 5))
        contour = ax.contour(xx, yy, Z, levels=10, cmap='viridis', linewidths=1.5)
        ax.clabel(contour, inline=True, fontsize=8)

        scatter = ax.scatter(
            X_pair[:, 0], X_pair[:, 1],
            c=y_reg, cmap='viridis', edgecolor='k', s=30
        )
        fig.colorbar(scatter, ax=ax, label="Actual Value")

        ax.set_xlabel(f1)
        ax.set_ylabel(f2)
        ax.set_title(f"KNN Surface: {f1} vs {f2}")
        figs.append(fig)

    return figs