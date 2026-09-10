# Spotify-Streaming-Data-and-Audio-Feature-2020-25-
This dataset analyzes Spotify streaming data and audio features from 2020–2025. It explores song popularity, streams, artists, genres, and features such as danceability, energy, valence, tempo, and acousticness to identify trends and understand factors influencing music popularity over time.
# ============================================
# Spotify Streaming Data and Audio Features
# 2020–2025
# ============================================

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load dataset
df = pd.read_csv("Spotify-Streaming-Data-and-Audio-Feature-2020-25.csv")

# 3. Display basic information
print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe(include="all"))

# ============================================
# 4. Remove duplicate records
# ============================================

df = df.drop_duplicates()

# ============================================
# 5. Handle missing values
# ============================================

numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ============================================
# 6. Correlation between audio features
# ============================================

audio_features = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness"
]

available_features = [col for col in audio_features if col in df.columns]

if available_features:
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        df[available_features].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Correlation Between Spotify Audio Features")
    plt.tight_layout()
    plt.show()

# ============================================
# 7. Distribution of audio features
# ============================================

if available_features:
    df[available_features].hist(
        figsize=(12, 8),
        bins=20,
        color="mediumseagreen"
    )
    plt.suptitle("Distribution of Spotify Audio Features")
    plt.tight_layout()
    plt.show()

# ============================================
# 8. Top songs by streams
# ============================================

# Change 'streams' if your dataset uses another column name
if "streams" in df.columns:
    top_songs = df.sort_values(
        by="streams",
        ascending=False
    ).head(10)

    print("\nTop 10 Most Streamed Songs:")
    print(top_songs)

# ============================================
# 9. Top artists by total streams
# ============================================

if "artist" in df.columns and "streams" in df.columns:

    artist_streams = (
        df.groupby("artist")["streams"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(12, 6))
    artist_streams.sort_values().plot(
        kind="barh",
        color="purple"
    )

    plt.title("Top 10 Artists by Total Streams")
    plt.xlabel("Total Streams")
    plt.ylabel("Artist")
    plt.tight_layout()
    plt.show()

# ============================================
# 10. Popularity analysis
# ============================================

if "popularity" in df.columns:

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["popularity"],
        bins=20,
        kde=True,
        color="royalblue"
    )

    plt.title("Distribution of Song Popularity")
    plt.xlabel("Popularity")
    plt.ylabel("Number of Songs")
    plt.show()

# ============================================
# 11. Popularity vs Danceability
# ============================================

if "popularity" in df.columns and "danceability" in df.columns:

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="danceability",
        y="popularity",
        alpha=0.6,
        color="green"
    )

    plt.title("Popularity vs Danceability")
    plt.xlabel("Danceability")
    plt.ylabel("Popularity")
    plt.show()

# ============================================
# 12. Popularity vs Energy
# ============================================

if "popularity" in df.columns and "energy" in df.columns:

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="energy",
        y="popularity",
        alpha=0.6,
        color="red"
    )

    plt.title("Popularity vs Energy")
    plt.xlabel("Energy")
    plt.ylabel("Popularity")
    plt.show()

# ============================================
# 13. Popularity vs Valence
# ============================================

if "popularity" in df.columns and "valence" in df.columns:

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="valence",
        y="popularity",
        alpha=0.6,
        color="orange"
    )

    plt.title("Popularity vs Valence")
    plt.xlabel("Valence")
    plt.ylabel("Popularity")
    plt.show()

# ============================================
# 14. Audio feature comparison
# ============================================

if available_features:

    mean_features = df[available_features].mean()

    plt.figure(figsize=(10, 6))

    mean_features.plot(
        kind="bar",
        color="teal"
    )

    plt.title("Average Spotify Audio Features")
    plt.xlabel("Audio Feature")
    plt.ylabel("Average Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ============================================
# 15. Year-wise streaming analysis
# ============================================

# Automatically detect a year/date column
year_column = None

for col in ["year", "Year", "release_year", "Release Year"]:
    if col in df.columns:
        year_column = col
        break

if year_column and "streams" in df.columns:

    yearly_streams = (
        df.groupby(year_column)["streams"]
        .sum()
    )

    plt.figure(figsize=(10, 6))

    yearly_streams.plot(
        marker="o",
        linewidth=2,
        color="darkblue"
    )

    plt.title("Total Spotify Streams by Year")
    plt.xlabel("Year")
    plt.ylabel("Total Streams")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ============================================
# 16. Year-wise average popularity
# ============================================

if year_column and "popularity" in df.columns:

    yearly_popularity = (
        df.groupby(year_column)["popularity"]
        .mean()
    )

    plt.figure(figsize=(10, 6))

    yearly_popularity.plot(
        kind="bar",
        color="coral"
    )

    plt.title("Average Song Popularity by Year")
    plt.xlabel("Year")
    plt.ylabel("Average Popularity")
    plt.tight_layout()
    plt.show()

# ============================================
# 17. Save cleaned dataset
# ============================================

df.to_csv(
    "Spotify_Cleaned_2020_2025.csv",
    index=False
)

print("\nAnalysis completed successfully!")
print("Cleaned dataset saved as Spotify_Cleaned_2020_2025.csv")
