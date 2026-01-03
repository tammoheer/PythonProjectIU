from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sqlalchemy import create_engine
from bokeh.plotting import figure, output_file, save


class ProjectError(Exception):
    """Base class for project-specific errors."""


class DataShapeError(ProjectError):
    """Raised when the CSV file structure is not as expected."""


# Data loading and validation logic

class DataLoader:
    """Loads train/ideal/test CSV files as DataFrames and validates their structure."""

    REQUIRED_TRAIN_PREFIX = "y"
    REQUIRED_TEST_COLS = ["x", "y"]

    def __init__(self, train_path: str | Path, ideal_path: str | Path, test_path: str | Path):
        self.train_path = Path(train_path)
        self.ideal_path = Path(ideal_path)
        self.test_path = Path(test_path)

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            train = pd.read_csv(self.train_path)
            ideal = pd.read_csv(self.ideal_path)
            test = pd.read_csv(self.test_path)
        except FileNotFoundError as e:
            raise ProjectError(f"File not found: {e.filename}") from e
        except Exception as e:
            raise ProjectError(f"Error while reading CSV files: {e}") from e

        # Basic validation
        if "x" not in train.columns or "x" not in ideal.columns:
            raise DataShapeError("train.csv and ideal.csv must contain an 'x' column.")
        if any(c not in test.columns for c in self.REQUIRED_TEST_COLS):
            raise DataShapeError("test.csv must contain the columns ['x','y'].")

        # Expected structure (IU): train: x, y1..y4 ; ideal: x, y1..y50
        train_ycols = [c for c in train.columns if c.lower().startswith("y")]
        if len(train_ycols) != 4:
            raise DataShapeError("train.csv should contain exactly 4 y-columns (y1..y4).")

        ideal_ycols = [c for c in ideal.columns if c.lower().startswith("y")]
        if len(ideal_ycols) < 4:
            raise DataShapeError("ideal.csv should contain many y-columns (e.g., y1..y50).")

        # Sort by x and ensure consistent types
        train = train.sort_values("x").reset_index(drop=True)
        ideal = ideal.sort_values("x").reset_index(drop=True)
        test = test.sort_values("x").reset_index(drop=True)

        return train, ideal, test


# OOP: Base class + inheritance
@dataclass
class BaseProcessor:
    train: pd.DataFrame
    ideal: pd.DataFrame

    @property
    def train_ycols(self) -> List[str]:
        return [c for c in self.train.columns if c.lower().startswith("y")]

    @property
    def ideal_ycols(self) -> List[str]:
        return [c for c in self.ideal.columns if c.lower().startswith("y")]


class FunctionSelector(BaseProcessor):
    """Selects for each training function the best matching ideal function (Least Squares)."""

    def select_best(self) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Returns:
            mapping_train_to_ideal: e.g., {'y1':'y17', 'y2':'y3', ...}
            max_abs_dev_per_train: e.g., {'y1': 0.42, ...}  (max |train - ideal| per training function)
        """
        # Ensure x-values are identical
        if not self.train["x"].equals(self.ideal["x"]):
            raise DataShapeError("The 'x' columns in train and ideal must be identical (no interpolation mode implemented).")

        mapping: Dict[str, str] = {}
        max_dev: Dict[str, float] = {}

        for ty in self.train_ycols:
            sse_min = math.inf
            best_iy = None
            for iy in self.ideal_ycols:
                diff = self.train[ty] - self.ideal[iy]
                sse = float((diff ** 2).sum())
                if sse < sse_min:
                    sse_min, best_iy = sse, iy

            assert best_iy is not None
            mapping[ty] = best_iy
            max_dev[ty] = float((self.train[ty] - self.ideal[best_iy]).abs().max())

        return mapping, max_dev


class TestMapper(BaseProcessor):
    """Assigns test points to the selected ideal functions (threshold according to IU: √2 * maxTrainDeviation)."""

    def __init__(self, train: pd.DataFrame, ideal: pd.DataFrame, test: pd.DataFrame):
        super().__init__(train, ideal)
        self.test = test

    def map_points(
        self,
        mapping_train_to_ideal: Dict[str, str],
        max_abs_dev_per_train: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns ['x','y','delta_y','ideal_func'].
        Only points that meet the threshold rule are included.
        """
        if not self.train["x"].equals(self.ideal["x"]):
            raise DataShapeError("The 'x' columns in train and ideal must be identical.")
        # Map: ideal_func -> corresponding training function (for threshold)
        ideal_to_train = {iy: ty for ty, iy in mapping_train_to_ideal.items()}

        # Merge test data with the four selected ideal functions on 'x'
        chosen_ideal_cols = list(ideal_to_train.keys())
        ideal_subset = self.ideal[["x"] + chosen_ideal_cols]
        merged = pd.merge(self.test, ideal_subset, on="x", how="left")

        rows = []
        for _, row in merged.iterrows():
            y_test = row["y"]
            # Minimum deviation among the four selected ideal functions
            best_iy, best_delta = None, math.inf
            for iy in chosen_ideal_cols:
                ideal_y = row[iy]
                delta = abs(y_test - ideal_y)
                if delta < best_delta:
                    best_delta, best_iy = delta, iy

            assert best_iy is not None
            # Threshold: √2 * maxTrainDeviation (for corresponding training function)
            train_for_iy = ideal_to_train[best_iy]
            threshold = math.sqrt(2.0) * max_abs_dev_per_train[train_for_iy]

            if best_delta <= threshold:
                rows.append(
                    {"x": row["x"], "y": y_test, "delta_y": float(best_delta), "ideal_func": best_iy}
                )

        return pd.DataFrame(rows, columns=["x", "y", "delta_y", "ideal_func"])


# -----------------------------
# Persistence (SQLite via SQLAlchemy)
# -----------------------------
class SQLiteRepository:
    """Stores DataFrames in an SQLite database (tables: train, ideal, mapping)."""

    def __init__(self, db_path: str | Path = "results.db"):
        self.engine = create_engine(f"sqlite:///{Path(db_path)}", echo=False)

    def save_train(self, df: pd.DataFrame):
        df.to_sql("train", self.engine, if_exists="replace", index=False)

    def save_ideal(self, df: pd.DataFrame):
        df.to_sql("ideal", self.engine, if_exists="replace", index=False)

    def save_mapping(self, df: pd.DataFrame):
        df.to_sql("mapping", self.engine, if_exists="replace", index=False)


# -----------------------------
# Visualization (Bokeh)
# -----------------------------
class Visualizer:
    """Creates two HTML plots: (1) Training vs. selected ideals, (2) Test point mappings."""

    def __init__(self, out_dir: str | Path = "plots"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def plot_train_vs_ideal(
        self,
        train: pd.DataFrame,
        ideal: pd.DataFrame,
        mapping_train_to_ideal: Dict[str, str],
        output_name: str = "train_vs_ideal.html",
    ) -> Path:
        p = figure(title="Train vs. Selected Ideal Functions", x_axis_label="x", y_axis_label="y", width=900, height=500)

        # Training points (circles)
        for ty in [c for c in train.columns if c.lower().startswith("y")]:
            p.circle(train["x"], train[ty], legend_label=f"train {ty}", size=5, alpha=0.6)

        # Selected ideal functions (lines)
        for ty, iy in mapping_train_to_ideal.items():
            p.line(ideal["x"], ideal[iy], legend_label=f"ideal for {ty}: {iy}", line_width=2)

        p.legend.click_policy = "hide"
        out_path = self.out_dir / output_name
        output_file(out_path)
        save(p)
        return out_path

    def plot_test_mapping(
        self,
        ideal: pd.DataFrame,
        mapping_df: pd.DataFrame,
        output_name: str = "test_mapping.html",
    ) -> Path:
        p = figure(title="Mapped Test Points to Ideal Functions", x_axis_label="x", y_axis_label="y", width=900, height=500)

        # Ideal function lines for the four selected functions
        chosen_iy = sorted(mapping_df["ideal_func"].unique())
        subset = ideal[["x"] + chosen_iy]
        for iy in chosen_iy:
            p.line(subset["x"], subset[iy], legend_label=f"ideal {iy}", line_width=2)

        # Assigned test points (squares)
        p.square(mapping_df["x"], mapping_df["y"], legend_label="mapped test points", size=6, alpha=0.8)

        p.legend.click_policy = "hide"
        out_path = self.out_dir / output_name
        output_file(out_path)
        save(p)
        return out_path


# -----------------------------
# Orchestration
# -----------------------------
def run_pipeline(train_csv: str | Path, ideal_csv: str | Path, test_csv: str | Path, db_path: str | Path = "results.db"):
    # 1) Load data
    loader = DataLoader(train_csv, ideal_csv, test_csv)
    train, ideal, test = loader.load()

    # 2) Select four ideal functions + calculate max deviation
    selector = FunctionSelector(train=train, ideal=ideal)
    mapping_train_to_ideal, max_abs_dev_per_train = selector.select_best()

    # 3) Map test points
    mapper = TestMapper(train=train, ideal=ideal, test=test)
    mapping_df = mapper.map_points(mapping_train_to_ideal, max_abs_dev_per_train)

    # 4) Save results to SQLite
    repo = SQLiteRepository(db_path=db_path)
    repo.save_train(train)
    repo.save_ideal(ideal)
    repo.save_mapping(mapping_df)

    # 5) Visualization
    vis = Visualizer(out_dir="plots")
    plot1 = vis.plot_train_vs_ideal(train, ideal, mapping_train_to_ideal, "train_vs_ideal.html")
    plot2 = vis.plot_test_mapping(ideal, mapping_df, "test_mapping.html")

    # Console summary
    print("=== Training Function -> Ideal Function Selection ===")
    for ty, iy in mapping_train_to_ideal.items():
        print(f"{ty:>3}  ->  {iy:>4}   (max |Δ| = {max_abs_dev_per_train[ty]:.6f})")

    print(f"\nSaved to SQLite: {Path(db_path).resolve()}")
    print(f"Mapping entries: {len(mapping_df)}")
    print(f"Plots: {plot1.resolve()}, {plot2.resolve()}")

    return mapping_train_to_ideal, max_abs_dev_per_train, mapping_df


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="IU DLMDSPWP01 Pipeline")
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--ideal", required=True, help="Path to ideal.csv")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument("--db", default="results.db", help="SQLite output file")
    parser.add_argument("--plots", default="plots", help="Directory for HTML plots")


    args = parser.parse_args()

    # Execute pipeline
    mapping_train_to_ideal, max_abs_dev_per_train, mapping_df = run_pipeline(
        train_csv=Path(args.train),
        ideal_csv=Path(args.ideal),
        test_csv=Path(args.test),
        db_path=Path(args.db),
    )
