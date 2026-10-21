from pathlib import Path
import pandas as pd

# Find the project folder and dataset
project_root = Path(__file__).resolve().parent.parent
dataset_path = (
    project_root / "data" / "raw" / "train-00000-of-00001.parquet"
)

print("=" * 55)
print("SKILLGRAPH-X DATASET INSPECTION")
print("=" * 55)

if not dataset_path.exists():
    print(f"ERROR: Dataset not found at {dataset_path}")
    raise SystemExit(1)

df = pd.read_parquet(dataset_path)

print("\nDataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst five records:")
print(df.head().to_string(index=False))

print("\nInspection completed!")


