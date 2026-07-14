from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=697)
df = dataset.data.original
df.to_csv("data/raw/dropout.csv", index=False)

print(f"Dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
