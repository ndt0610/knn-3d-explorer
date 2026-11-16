# 3D k-NN Explorer

An interactive Streamlit tool for inspecting k-nearest-neighbor classification on user-entered three-dimensional points.

## What it demonstrates

- Euclidean distance ranking in three dimensions.
- Majority voting among the selected k nearest points.
- A deterministic tie rule.
- A Plotly 3D view of labeled points, the query, and a separate radius sphere.

The radius control is a visualization aid. It does not change the k-NN prediction, which is based only on the nearest k examples.

## Run

    python -m pip install -r requirements.txt
    streamlit run knn_3d_manual_app.py

The default dataset has six points, k=3, and query (4, 5, 6). Every supplied row must contain exactly three finite numeric coordinates.

## Check

    python -m unittest discover -s tests -v

- knn_3d_manual_app.py — input validation, distance calculation, prediction, and chart.
- tests/test_knn.py — nearest-neighbor, tie, and input validation checks.

This is an algorithm illustration on a tiny manual dataset, not a measured classification benchmark.
