# US Honey Exploratory Data Analysis

Brief EDA of the US honey dataset using Plotly and Dash (inline notebook + small Dash apps).

Contents:
- `US_HONEY.ipynb` — main notebook with data loading, cleaning checks, and interactive Plotly visualizations (line, scatter, box, heatmap, sunburst, choropleth).
- `US_honey_dataset (1) (1).csv` — source data file (state-level honey metrics by year).


Quick start:
1. Open `US_HONEY.ipynb` in Jupyter / VS Code and run all cells.
2. If Dash apps show deprecation warnings, install latest `dash`:

```bash
pip install dash jupyter-dash
```

Using `uv` (if you have the `uv` CLI available):

```powershell
uv add dash plotly pandas jupyter-dash
```

Or on a POSIX shell:

```bash
uv add dash plotly pandas jupyter-dash
```

Notes:
- The notebook uses inline Dash rendering (`app.run(mode='inline')`) for interactive controls. If inline rendering fails, run the Dash app cells as standalone scripts and open the shown local URL.
- Visual styling uses `plotly_dark`/`plotly_white` templates depending on the chart.

Contact:
- For further interpretation or additional plots, open an issue or message the author in-repo.
