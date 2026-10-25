
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Project setup
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 2. Load Netflix dataset
df = pd.read_csv(BASE_DIR / "Dataset.csv")
df.columns = df.columns.str.strip()

required = {"show_id", "type", "rating", "listed_in"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Dataset is missing columns: {missing}")

df = df.drop_duplicates().copy()

print("=" * 55)
print("AUSPIFY TASK 5: CONTENT RATING & GENRE ANALYSIS")
print("=" * 55)
print("Dataset loaded:", df.shape)

# 3. Clean rating and genre information
df["rating"] = df["rating"].fillna("Not Rated").astype(str).str.strip()
df["listed_in"] = df["listed_in"].fillna("").astype(str).str.strip()
df["type"] = df["type"].fillna("Unknown").astype(str).str.strip()

# 4. Analyze content rating categories
rating_counts = (
    df.groupby(["rating", "type"])
    .size()
    .unstack(fill_value=0)
)

rating_counts["Total"] = rating_counts.sum(axis=1)
rating_counts = rating_counts.sort_values("Total", ascending=False)
rating_counts.to_csv(OUTPUT_DIR / "rating_analysis.csv")

print("\nCONTENT RATINGS")
print(rating_counts.to_string())

# 5. Analyze genres
genre_data = df[df["listed_in"] != ""].copy()
genre_data["genre"] = genre_data["listed_in"].str.split(",")
genre_data = genre_data.explode("genre")
genre_data["genre"] = genre_data["genre"].str.strip()
genre_data = genre_data[genre_data["genre"] != ""]

genre_data = genre_data.drop_duplicates(
    subset=["show_id", "genre"]
)

genre_counts = (
    genre_data.groupby("genre")["show_id"]
    .nunique()
    .sort_values(ascending=False)
    .rename("title_count")
    .reset_index()
)

genre_counts.to_csv(OUTPUT_DIR / "genre_analysis.csv", index=False)

print("\nTOP 10 GENRES")
print(genre_counts.head(10).to_string(index=False))

# 6. Compare rating categories by content type
rating_plot = (
    df.groupby(["rating", "type"])
    .size()
    .unstack(fill_value=0)
)

rating_plot = rating_plot.loc[
    rating_counts.head(10).index
]

rating_plot.plot(kind="bar", figsize=(12, 6))
plt.title("Top Content Rating Categories by Type")
plt.xlabel("Content Rating")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rating_comparison.png", dpi=300)
plt.close()

# 7. Visualize top 10 genres
top_genres = genre_counts.head(10).sort_values("title_count")

plt.figure(figsize=(12, 7))
plt.barh(top_genres["genre"], top_genres["title_count"])
plt.title("Top 10 Netflix Genres")
plt.xlabel("Number of Titles")
plt.ylabel("Genre")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_10_genres.png", dpi=300)
plt.close()

# 8. Compare genres across Movies and TV Shows
genre_type = (
    genre_data.groupby(["genre", "type"])["show_id"]
    .nunique()
    .unstack(fill_value=0)
)

genre_type["Total"] = genre_type.sum(axis=1)
genre_type = genre_type.sort_values("Total", ascending=False)

genre_type.to_csv(OUTPUT_DIR / "genre_by_content_type.csv")

top_genre_types = genre_type.head(10).drop(
    columns=["Total"]
)

top_genre_types.plot(kind="bar", figsize=(13, 7))
plt.title("Genre Distribution: Movies vs TV Shows")
plt.xlabel("Genre")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "genre_type_comparison.png", dpi=300)
plt.close()

# 9. Generate business insights
most_common_rating = rating_counts["Total"].idxmax()
most_common_genre = genre_counts.iloc[0]["genre"]

with open(
    OUTPUT_DIR / "business_insights.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("AUSPIFY TASK 5: BUSINESS INSIGHTS\n\n")
    file.write(
        f"Most common content rating category: "
        f"{most_common_rating}\n"
    )
    file.write(
        f"Most represented genre: {most_common_genre}\n\n"
    )
    file.write("TOP 10 GENRES:\n")

    for _, row in genre_counts.head(10).iterrows():
        file.write(
            f"{row['genre']}: {row['title_count']} titles\n"
        )

    file.write("\nINTERPRETATION:\n")
    file.write(
        "Content ratings describe audience suitability, "
        "not review scores or popularity ratings.\n"
    )
    file.write(
        "Genre counts can overlap because one title may "
        "belong to multiple genres.\n"
    )
    file.write(
        "These results describe the available dataset and "
        "do not prove audience preferences or viewing behavior.\n"
    )

# 10. Finish
print("\nANALYSIS COMPLETED SUCCESSFULLY!")
print("Output folder:", OUTPUT_DIR.resolve())

for item in sorted(OUTPUT_DIR.iterdir()):
    print("-", item.name)
