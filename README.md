# 3D k-NN Explorer

A small **Tkinter desktop demo** of k-nearest-neighbor classification on editable 3D points. It shows how each observation is ranked by Euclidean distance and how the selected *k* neighbors vote for the query point.

![Tkinter interface with two point groups, query coordinates, k setting, distance ranking and 3D scatter plot.](docs/images/knn-3d-classification.png)

*The screenshot uses the built-in points, query `(2.5, 3.5, 4.5)`, and `k = 3`. The nearest three points all belong to Class 1; the green table rows and 3D plot show the same result.*

## What the demo makes visible

- Add or edit coordinates for two labeled point groups.
- Rank every sample by Euclidean distance to the query point.
- Predict from the nearest *k* samples; ties follow the nearest selected neighbor.
- Inspect the points, query, radius sphere and in-sphere connections in the 3D plot.

The radius is a visualization aid only; it does not change the k-NN vote.

## Run locally

```bash
python -m pip install -r requirements.txt
python knn_3d_manual_app.py
```

## Project files

- `knn_3d_manual_app.py` — Tkinter controls, distance table and Matplotlib 3D view.
- `tests/` — checks for nearest-neighbor ranking and tie behavior.

This is a small learning demo. Its default points are hand-authored examples, not a trained financial model or evidence of general classification accuracy.
