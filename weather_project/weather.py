# ----- TASK 1: Data Acquisition and Loading -----

import pandas as pd

file_path = "GlobalWeatherRepository.csv"
df = pd.read_csv(file_path)

print("First 5 rows:")
print(df.head())

print("\nDataFrame Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ----- TASK 2: Data Cleaning and Processing -----

date_col = None
for col in df.columns:
    if df[col].dtype == "object":
        try:
            temp = pd.to_datetime(df[col], errors='coerce')
            if temp.notna().sum() > 0:
                df[col] = temp
                date_col = col
                break
        except:
            pass

if date_col is None:
    raise ValueError(" No valid datetime column detected. Check dataset.")

print(f"\n Datetime column detected: {date_col}")

# Handle missing numeric values
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

df = df.ffill()

# Select relevant columns
columns_to_keep = []
for col in df.columns:
    if any(keyword in col.lower() for keyword in ["temp", "temperature", "humidity", "rain", "precip", "wind", "date"]):
        columns_to_keep.append(col)

df_clean = df[columns_to_keep].copy()

print("\n Columns kept after filtering:")
print(df_clean.columns)

print("\n Cleaned Dataset Preview:")
print(df_clean.head())

# ------ TASK 3: Statistical Analysis with NumPy -----

import numpy as np

df_stats = df_clean.copy()
df_stats["Day"] = df_stats["last_updated"].dt.date
df_stats["Month"] = df_stats["last_updated"].dt.month
df_stats["Year"] = df_stats["last_updated"].dt.year

numeric_cols = df_stats.select_dtypes(include="number").columns

daily_stats = df_stats.groupby("Day")[numeric_cols].agg(["mean", "min", "max", "std"])
print("\n-----DAILY STATISTICS ------")
print(daily_stats)


monthly_stats = df_stats.groupby("Month")[numeric_cols].agg(["mean", "min", "max", "std"])
print("\n------MONTHLY STATISTICS ------")
print(monthly_stats)

yearly_stats = df_stats.groupby("Year")[numeric_cols].agg(["mean", "min", "max", "std"])
print("\n-----YEARLY STATISTICS ----")
print(yearly_stats)


print("\n NUMPY SUMMARY (Across Entire Dataset):")
for col in df_stats.select_dtypes(include=[np.number]).columns:
    col_values = df_stats[col].values
    print(f"\n➡ {col}")
    print("  Mean:", np.mean(col_values))
    print("  Min:", np.min(col_values))
    print("  Max:", np.max(col_values))
    print("  Std:", np.std(col_values))

# ----- TASK 4: Visualization with Matplotlib -------

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

if "temperature_celsius" in df_stats.columns:
    temp_col = "temperature_celsius"
elif "temp_c" in df_stats.columns:
    temp_col = "temp_c"
else:
    temp_col = [col for col in df_stats.columns if "temp" in col.lower()][0]

daily_temp = df_stats.groupby("Day")[temp_col].mean()

plt.figure(figsize=(12, 5))
plt.plot(daily_temp.index, daily_temp.values)
plt.title("Daily Temperature Trend")
plt.xlabel("Day")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.show()

# Detect rainfall column
rain_cols = [col for col in df_stats.columns if "rain" in col.lower() or "precip" in col.lower()]
rain_col = rain_cols[0]

monthly_rain = df_stats.groupby("Month")[rain_col].sum()

plt.figure(figsize=(12, 5))
plt.bar(monthly_rain.index.astype(str), monthly_rain.values)
plt.title("Monthly Rainfall Total")
plt.xlabel("Month")
plt.ylabel("Rainfall")
plt.xticks(rotation=45)
plt.show()

hum_cols = [col for col in df_stats.columns if "humid" in col.lower()]
humidity_col = hum_cols[0]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_stats, x=temp_col, y=humidity_col)
plt.title("Humidity vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.show()

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(daily_temp.index, daily_temp.values)
plt.title("Daily Temperature Trend")
plt.xlabel("Day")
plt.ylabel("Temperature")

plt.subplot(1, 2, 2)
sns.scatterplot(data=df_stats, x=temp_col, y=humidity_col)
plt.title("Humidity vs Temperature")
plt.xlabel("Temperature")
plt.ylabel("Humidity")

plt.tight_layout()
plt.show()

# ----- TASK 5: Grouping and Aggregation -----

print("\n----- TASK 5: Grouping and Aggregation -----")


df_stats["Month"] = df_stats[date_col].dt.month
df_stats["Season"] = df_stats["Month"] % 12 // 3 + 1

numeric_cols = df_stats.select_dtypes(include="number").columns
seasonal_group = df_stats.groupby("Season")[numeric_cols].agg(["mean", "min", "max", "std"]).reset_index()

print("\nSEASONAL AGGREGATE STATISTICS:")
print(seasonal_group.head())

df_resample = df_stats.set_index(date_col)

weekly_avg = df_resample.resample("W")[numeric_cols].mean()
print("\nWEEKLY AVERAGE VALUES (Resample):")
print(weekly_avg.head())

monthly_rain_resampled = df_resample[rain_col].resample("ME").sum()
print("\nMONTHLY RAINFALL TOTALS (Resample):")
print(monthly_rain_resampled.head())


# ----- TASK 6: Export & Storytelling -----

import os
output_dir = "weather_outputs"
os.makedirs(output_dir, exist_ok=True)

clean_csv_path = os.path.join(output_dir, "cleaned_weather_data.csv")
df_clean.to_csv(clean_csv_path, index=False)
print(f"\nCleaned CSV saved at: {clean_csv_path}")

plt.figure(figsize=(12, 5))
plt.plot(daily_temp.index, daily_temp.values)
plt.title("Daily Temperature Trend")
plt.xlabel("Day")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, "daily_temperature_trend.png"))
plt.close()

plt.figure(figsize=(12, 5))
plt.bar(monthly_rain.index.astype(str), monthly_rain.values)
plt.title("Monthly Rainfall Total")
plt.xlabel("Month")
plt.ylabel("Rainfall")
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, "monthly_rainfall.png"))
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_stats, x=temp_col, y=humidity_col)
plt.title("Humidity vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.savefig(os.path.join(output_dir, "humidity_vs_temperature.png"))
plt.close()

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(daily_temp.index, daily_temp.values)
plt.title("Daily Temperature Trend")
plt.xlabel("Day")
plt.ylabel("Temperature")

plt.subplot(1, 2, 2)
sns.scatterplot(data=df_stats, x=temp_col, y=humidity_col)
plt.title("Humidity vs Temperature")
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "combined_figure.png"))
plt.close()

print("All plots saved as PNG images.")

report_path = os.path.join(output_dir, "Weather_Report.md")

with open(report_path, "w", encoding="utf-8") as report:
    report.write("#  Weather Data Analysis Report\n\n")
    report.write("##  Project Summary\n")
    report.write("This project analyzes real-world weather data to understand temperature, rainfall, and humidity patterns using Python.\n\n")

    report.write("## Key Insights\n")
    report.write(f"- Total records analyzed: **{len(df_clean)}**\n")
    report.write(f"- Temperature column used: **{temp_col}**\n")
    report.write(f"- Date range: **{df_stats[date_col].min().date()} → {df_stats[date_col].max().date()}**\n\n")

    report.write("## Temperature Trends\n")
    report.write("- Daily temperatures fluctuate across the dataset.\n")
    report.write("- Line plot saved as: `daily_temperature_trend.png`\n\n")

    report.write("## Rainfall Patterns\n")
    report.write("- Monthly rainfall varies significantly.\n")
    report.write("- Bar chart saved as: `monthly_rainfall.png`\n\n")

    report.write("## Humidity & Temperature Relationship\n")
    report.write("- Scatter plot shows correlation between humidity and temperature.\n")
    report.write("- Saved as: `humidity_vs_temperature.png`\n\n")

    report.write("## Seasonal Behavior\n")
    report.write("- Grouping by seasons shows how weather varies across Winter, Spring, Summer, and Autumn.\n\n")

    report.write("##  Output Directory\n")
    report.write("All exported assets are saved in the **weather_outputs/** folder:\n")
    report.write("- Cleaned dataset CSV\n")
    report.write("- 4 PNG visualizations\n")
    report.write("- Markdown report\n")

print(f"Report generated at: {report_path}")








