# HomePricePredictor

Predicts house sale prices (`SalePrice`) from property features using gradient boosting. Notebook: `Untitled.ipynb`.

## Data
Kaggle-style house prices data (`train.csv`: 1,460 rows x 81 columns, plus `test.csv`). The CSVs are not in this repo; place them next to the notebook.

## Approach
1. **Cleaning:** drop `Id` and the sparse columns `Alley`, `PoolQC`, `Fence`, `MiscFeature`; fill numeric gaps with the median.
2. **Feature selection:** rank features by correlation with `SalePrice` (top: `OverallQual` 0.79, `GrLivArea` 0.71, `GarageCars` 0.64) and keep a hand-picked set of 21 features.
3. **Outliers:** remove z-score outliers (|z| > 3) on `GrLivArea` and `SalePrice` (1,460 -> 1,429 rows).
4. **Encoding:** label-encode the categorical columns (`Neighborhood`, `BldgType`, `HouseStyle`, `ExterQual`, `KitchenQual`).
5. **Model:** `GradientBoostingRegressor` tuned with `RandomizedSearchCV` (20 iterations, 5-fold CV, R² scoring).

## Results
Best cross-validated R²: **0.889** (`n_estimators=450`, `max_depth=2`, `learning_rate≈0.073`, `min_samples_split=5`, `min_samples_leaf=1`). The notebook writes `submission.csv` with 1,459 predictions.

## Run it
```bash
pip install pandas numpy scikit-learn scipy matplotlib jupyter
jupyter notebook
```

## Possible improvements
- Use a train/validation split and report RMSE, not just CV R².
