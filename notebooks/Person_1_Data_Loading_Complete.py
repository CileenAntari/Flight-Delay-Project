
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: f'{x:.2f}')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("[OK] Libraries imported successfully\n")

# ============================================================================
# INTRODUCTION
# ============================================================================

print("="*70)
print("FLIGHT DELAY PREDICTION - DATA LOADING AND EXPLORATORY ANALYSIS")
print("="*70)
print("\nThis notebook encompasses Person 1's tasks:")
print("  • Loading and inspecting the flight delays dataset")
print("  • Performing comprehensive exploratory data analysis (EDA)")
print("  • Identifying data quality issues and patterns")
print("  • Preparing the dataset for further processing")
print("\n")

# ============================================================================
# DATA SOURCE EXPLANATION
# ============================================================================

print("="*70)
print("DATA SOURCE EXPLANATION")
print("="*70)

print("""
Data Source: Kaggle
  • Repository: Flight Delay Dataset
  • Type: Real-world operational flight data

Airlines Included: Major U.S. carriers operating domestic flights

Airports Included (5 Major U.S. Hubs):
  • ATL - Hartsfield-Jackson Atlanta International
  • DFW - Dallas/Fort Worth International
  • JFK - John F. Kennedy International
  • LAX - Los Angeles International
  • ORD - Chicago O'Hare International

Dataset Objective:
  To provide comprehensive flight operational data for analyzing delay patterns,
  understanding factors influencing flight punctuality, and developing predictive
  models for flight delay estimation.

Information Contained:
  • Flight identification and scheduling information
  • Departure and arrival times (scheduled vs. actual)
  • Airline and aircraft information
  • Airport codes and routes
  • Delay metrics in minutes
  • Delay reason classification
  • Flight status flags (cancelled, diverted)
""")

# ============================================================================
# LOAD DATASET
# ============================================================================

data_path = '../data/raw/flight_delays.csv'

try:
    df = pd.read_csv(data_path)
    print("[OK] Dataset loaded successfully!")
    print(f"  • Rows: {df.shape[0]:,}")
    print(f"  • Columns: {df.shape[1]:,}")
    print(f"  • Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
except FileNotFoundError:
    print(f"[ERROR] File not found at {data_path}")
    exit()

# Convert date columns to datetime format
date_columns = [
    'ScheduledDeparture',
    'ActualDeparture',
    'ScheduledArrival',
    'ActualArrival'
]

for col in date_columns:
    df[col] = pd.to_datetime(df[col])

print("[OK] Date columns converted to datetime format")

# Display dataset time coverage
print("\nDate Range:")
print(f"  • Start Date: {df['ScheduledDeparture'].min()}")
print(f"  • End Date:   {df['ScheduledDeparture'].max()}\n")

# ============================================================================
# DATASET DIMENSIONS
# ============================================================================

print("="*70)
print("DATASET DIMENSIONS")
print("="*70)

rows = df.shape[0]
cols = df.shape[1]
total_cells = rows * cols

print(f"\nDataset Size Information:")
print(f"  • Total Rows:   {rows:,}")
print(f"  • Total Columns: {cols:,}")
print(f"  • Total Cells:  {total_cells:,}")
print(f"  • Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

# ============================================================================
# COLUMN NAMES
# ============================================================================

print("="*70)
print("COLUMN NAMES")
print("="*70)

print("\nAll columns in the dataset:\n")
for idx, col_name in enumerate(df.columns, 1):
    print(f"  {idx:2d}. {col_name}")

print()

# ============================================================================
# DATA TYPE DISTRIBUTION
# ============================================================================

print("="*70)
print("DATA TYPE DISTRIBUTION")
print("="*70)

dtype_counts = df.dtypes.value_counts()

print("\nNumber of columns by data type:")
for dtype, count in dtype_counts.items():
    print(f"  • {str(dtype):20s}: {count} columns")

print()

# ============================================================================
# INITIAL DATA INSPECTION
# ============================================================================

print("="*70)
print("DATASET STRUCTURE")
print("="*70)

print("\nFirst 5 rows:")
print(df.head())

print("\n\nDataset Information:")
df.info()

# ============================================================================
# MISSING VALUES ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("MISSING VALUES ANALYSIS")
print("="*70)

missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing Count': df.isnull().sum().values,
    'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
})

missing_filtered = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing %', ascending=False)

if len(missing_filtered) > 0:
    print("\nColumns with Missing Values:")
    print(missing_filtered.to_string(index=False))
else:
    print("\n[OK] No missing values detected!")

completeness = (1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))) * 100
print(f"\nOverall Data Completeness: {completeness:.2f}%")

# ============================================================================
# DATASET DESCRIPTION TABLE
# ============================================================================

print("\n" + "="*70)
print("DATASET DESCRIPTION TABLE")
print("="*70)

description_table = pd.DataFrame({
    'Column Name': df.columns,
    'Data Type': df.dtypes.values,
    'Missing Count': df.isnull().sum().values,
    'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
})

print("\n")
print(description_table.to_string(index=False))
print()

# ============================================================================
# INTERPRETATION: Missing Values
# ============================================================================
print("\n" + "-"*70)
print("[INFO] MISSING VALUES INTERPRETATION:")
print("-"*70)
print("""
The analysis of missing values reveals the data quality and completeness level.
A completeness rate of {:.2f}% indicates that most records contain valid values
across all features. This level of completeness is favorable for exploratory
analysis and suggests minimal data quality issues at this stage.
""".format(completeness))

# ============================================================================
# DUPLICATE RECORDS
# ============================================================================

print("\n" + "="*70)
print("DUPLICATE RECORDS ANALYSIS")
print("="*70)

total_dups = df.duplicated().sum()
flightid_dups = df['FlightID'].duplicated().sum()

print(f"\nTotal Duplicate Rows: {total_dups}")
print(f"Duplicate FlightIDs: {flightid_dups}")

if total_dups == 0:
    print("[OK] No duplicates found")

# ============================================================================
# CATEGORICAL VARIABLES
# ============================================================================

print("\n" + "="*70)
print("CATEGORICAL VARIABLES - UNIQUE VALUES")
print("="*70)

categorical_cols = ['Airline', 'Origin', 'Destination', 'DelayReason', 'AircraftType']

for col in categorical_cols:
    unique_count = df[col].nunique()
    print(f"\n{col}:")
    print(f"  • Unique Values: {unique_count}")

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS - NUMERIC VARIABLES")
print("="*70)

numeric_cols = df.select_dtypes(include=[np.number]).columns
print("\nBasic Statistics:")
print(df[numeric_cols].describe())

# ============================================================================
# DETAILED DELAY ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("DETAILED ANALYSIS - DELAY MINUTES")
print("="*70)

delay_stats = df['DelayMinutes'].describe()

print("\nDelay Statistics (in minutes):")
print(f"  Count:   {delay_stats['count']:.0f} flights")
print(f"  Mean:    {delay_stats['mean']:.2f} minutes")
print(f"  Std Dev: {delay_stats['std']:.2f} minutes")
print(f"  Min:     {delay_stats['min']:.2f} minutes")
print(f"  25%:     {delay_stats['25%']:.2f} minutes")
print(f"  Median:  {delay_stats['50%']:.2f} minutes")
print(f"  75%:     {delay_stats['75%']:.2f} minutes")
print(f"  Max:     {delay_stats['max']:.2f} minutes")

early = (df['DelayMinutes'] < 0).sum()
ontime = (df['DelayMinutes'] == 0).sum()
delayed = (df['DelayMinutes'] > 0).sum()
significant = (df['DelayMinutes'] > 15).sum()

print(f"\nDelay Status Distribution:")
print(f"  • Early Arrivals:     {early:,} ({early/len(df)*100:.2f}%)")
print(f"  • On-Time (0 min):    {ontime:,} ({ontime/len(df)*100:.2f}%)")
print(f"  • Any Delay (>0 min): {delayed:,} ({delayed/len(df)*100:.2f}%)")
print(f"  • Significant (>15):  {significant:,} ({significant/len(df)*100:.2f}%)")

# ============================================================================
# OUTLIER DETECTION - DELAY MINUTES
# ============================================================================

print("\n" + "="*70)
print("OUTLIER DETECTION - DELAY MINUTES (IQR METHOD)")
print("="*70)

delay_data = df['DelayMinutes']
Q1 = delay_data.quantile(0.25)
Q3 = delay_data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = ((delay_data < lower_bound) | (delay_data > upper_bound)).sum()
outlier_percentage = (outliers / len(df)) * 100

print(f"\nIQR Calculation:")
print(f"  • Q1 (25th percentile):  {Q1:.2f} minutes")
print(f"  • Q3 (75th percentile):  {Q3:.2f} minutes")
print(f"  • IQR (Q3 - Q1):         {IQR:.2f} minutes")

print(f"\nBounds:")
print(f"  • Lower Bound: {lower_bound:.2f} minutes")
print(f"  • Upper Bound: {upper_bound:.2f} minutes")

print(f"\nOutlier Summary:")
print(f"  • Number of Outliers:    {outliers:,}")
print(f"  • Percentage of Outliers: {outlier_percentage:.2f}%")

# ============================================================================
# INTERPRETATION: Delay Analysis
# ============================================================================
print("\n" + "-"*70)
print("[INFO] DELAY ANALYSIS INTERPRETATION:")
print("-"*70)
print("""
The delay distribution provides insights into flight punctuality patterns.
The mean delay of {:.2f} minutes and median of {:.2f} minutes indicate the
typical delay experienced. The standard deviation of {:.2f} minutes shows
variability in delay magnitude. The presence of both negative values (early
arrivals) and positive values (delays) represents the full spectrum of
operational outcomes in the dataset.
""".format(delay_stats['mean'], delay_stats['50%'], delay_stats['std']))

# ============================================================================
# GEOGRAPHIC ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("GEOGRAPHIC ANALYSIS - AIRPORTS")
print("="*70)

origins = df['Origin'].unique()
destinations = df['Destination'].unique()
all_airports = sorted(set(list(origins) + list(destinations)))

print(f"\nUnique Airports: {len(all_airports)}")
print(f"Airport Codes: {', '.join(all_airports)}")

airport_mapping = {
    'ATL': 'Hartsfield-Jackson Atlanta International',
    'DFW': 'Dallas/Fort Worth International',
    'JFK': 'John F. Kennedy International',
    'LAX': 'Los Angeles International',
    'ORD': 'Chicago O\'Hare International'
}

print("\nAirport Activity:")
for airport in sorted(all_airports):
    name = airport_mapping.get(airport, 'Unknown')
    origin_count = (df['Origin'] == airport).sum()
    dest_count = (df['Destination'] == airport).sum()
    total = origin_count + dest_count
    print(f"\n{airport} - {name}")
    print(f"  • Departures: {origin_count:,}")
    print(f"  • Arrivals: {dest_count:,}")
    print(f"  • Total: {total:,}")

# ============================================================================
# INTERPRETATION: Airport Distribution
# ============================================================================
print("\n" + "-"*70)
print("[INFO] AIRPORT ANALYSIS INTERPRETATION:")
print("-"*70)
print("""
The geographic analysis across five major U.S. airports demonstrates the
spatial scope of the dataset. Each airport's activity is measured by both
departure and arrival frequencies. The variation in airport activity reflects
differences in airport size, hub status, and regional connectivity. Atlanta,
Dallas/Fort Worth, New York, Los Angeles, and Chicago represent the busiest
aviation hubs in the United States, making them suitable for understanding
national delay patterns.
""")

# ============================================================================
# AIRLINE ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("AIRLINE DISTRIBUTION")
print("="*70)

airline_counts = df['Airline'].value_counts()

print(f"\nTotal Airlines: {len(airline_counts)}\n")
print("Airline Breakdown:")

for airline, count in airline_counts.items():
    percentage = (count / len(df)) * 100
    bar_length = int(percentage / 2)
    bar = '#' * bar_length
    print(f"  {airline:20s} {count:8,} flights ({percentage:5.2f}%) {bar}")

print(f"\nTotal Flights: {airline_counts.sum():,}")

# ============================================================================
# INTERPRETATION: Airline Distribution
# ============================================================================
print("\n" + "-"*70)
print("[INFO] AIRLINE DISTRIBUTION INTERPRETATION:")
print("-"*70)
print("""
The airline distribution reveals the composition of carriers in the dataset.
With {:.0f} different airlines represented, the data encompasses multiple
operators. The variation in flight counts per airline reflects differences
in market share, operational volume, and network coverage among carriers
in the sample. This diversity is important for understanding delay patterns
across different airline operational models.
""".format(len(airline_counts)))

# ============================================================================
# DATA DISTRIBUTION ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("DATA DISTRIBUTION ANALYSIS")
print("="*70)

# Airline balance
airline_percentages = (airline_counts / len(df) * 100).values
max_airline_pct = airline_percentages.max()
min_airline_pct = airline_percentages.min()
airline_balance_diff = max_airline_pct - min_airline_pct

print(f"\n[STATS] AIRLINE DISTRIBUTION BALANCE:")
print(f"  • Maximum representation: {max_airline_pct:.2f}%")
print(f"  • Minimum representation: {min_airline_pct:.2f}%")
print(f"  • Difference: {airline_balance_diff:.2f}%")

if airline_balance_diff > 20:
    print(f"  ⚠ Assessment: Airlines are NOT equally represented.")
    print(f"    This suggests NATURAL variation (not stratified sampling).")
else:
    print(f"  [OK] Airlines show relatively balanced representation.")

# Airport balance
origin_counts = df['Origin'].value_counts()
origin_percentages = (origin_counts / len(df) * 100).values
origin_max = origin_percentages.max()
origin_min = origin_percentages.min()
origin_balance = origin_max - origin_min

dest_counts = df['Destination'].value_counts()
dest_percentages = (dest_counts / len(df) * 100).values
dest_max = dest_percentages.max()
dest_min = dest_percentages.min()
dest_balance = dest_max - dest_min

print(f"\n[STATS] AIRPORT DISTRIBUTION BALANCE:")
print(f"  Origins:")
print(f"    • Max: {origin_max:.2f}%  |  Min: {origin_min:.2f}%  |  Range: {origin_balance:.2f}%")
print(f"  Destinations:")
print(f"    • Max: {dest_max:.2f}%  |  Min: {dest_min:.2f}%  |  Range: {dest_balance:.2f}%")

# Target variable balance
def categorize_delay(minutes):
    if minutes < 0:
        return 'Early'
    elif minutes == 0:
        return 'On-Time'
    elif minutes <= 15:
        return 'Short Delay'
    elif minutes <= 60:
        return 'Medium Delay'
    else:
        return 'Long Delay'

delay_categories = df['DelayMinutes'].apply(categorize_delay)
delay_dist = delay_categories.value_counts()
delay_percentages = (delay_dist / len(df) * 100).values
delay_max_pct = delay_percentages.max()
delay_min_pct = delay_percentages.min()

print(f"\n[STATS] DELAY CATEGORY DISTRIBUTION (Target Variable):")
for category in ['Early', 'On-Time', 'Short Delay', 'Medium Delay', 'Long Delay']:
    if category in delay_dist.index:
        count = delay_dist[category]
        pct = (count / len(df)) * 100
        print(f"  • {category:15s}: {count:8,} ({pct:6.2f}%)")

print(f"\n  Balance assessment:")
print(f"    • Highest category: {delay_max_pct:.2f}%")
print(f"    • Lowest category: {delay_min_pct:.2f}%")

# Overall conclusion
print(f"\n[NOTE] DISTRIBUTION ASSESSMENT:\n")

imbalance_indicators = 0
if airline_balance_diff > 20:
    imbalance_indicators += 1
if origin_balance > 25:
    imbalance_indicators += 1
if dest_balance > 25:
    imbalance_indicators += 1
if delay_max_pct > 50:
    imbalance_indicators += 1

if imbalance_indicators >= 2:
    print("  The dataset shows natural variation across airlines and airports.")
    print("\n  Observations:")
    print("  • Airlines and airports show varying levels of representation")
    print("  • Distribution reflects differences in flight operations volume")
    print("  • Busier airlines and airports have higher frequency")
    print("  • Delay patterns vary across categorical variables")
    print("\n  Assessment: The data exhibits natural variation typical of")
    print("  operational flight records.")
else:
    print("  The dataset shows relatively consistent representation across")
    print("  airlines and airports.")

# ============================================================================
# INTERPRETATION: Data Distribution Analysis
# ============================================================================
print("\n" + "-"*70)
print("[INFO] DATA DISTRIBUTION INTERPRETATION:")
print("-"*70)
print("""
The distribution analysis across multiple dimensions (airlines, airports,
and delays) reveals the structure of the operational data. Understanding
these distributions is essential for recognizing which factors may influence
delay patterns. The variations observed across categories suggest that delay
outcomes are influenced by multiple operational and organizational factors,
which supports the need for comprehensive exploratory analysis in subsequent
phases of the project.
""")

# ============================================================================
# TARGET VARIABLE CANDIDATE
# ============================================================================

print("\n" + "="*70)
print("TARGET VARIABLE CANDIDATE")
print("="*70)

print("\nDelayMinutes was identified as the primary target variable")
print("for future delay prediction and punctuality analysis.")

# ============================================================================
# OPERATIONAL STATUS
# ============================================================================

print("\n" + "="*70)
print("FLIGHT OPERATIONAL STATUS")
print("="*70)

cancelled_count = df['Cancelled'].sum()
diverted_count = df['Diverted'].sum()
both_count = ((df['Cancelled'] == True) & (df['Diverted'] == True)).sum()
normal_count = len(df) - cancelled_count - diverted_count + both_count

print(f"\nFlight Status Summary:")
print(f"  • Normal Flights:           {normal_count:,} ({normal_count/len(df)*100:.2f}%)")
print(f"  • Cancelled Flights:        {cancelled_count:,} ({cancelled_count/len(df)*100:.2f}%)")
print(f"  • Diverted Flights:         {diverted_count:,} ({diverted_count/len(df)*100:.2f}%)")
print(f"  • Both Cancelled & Diverted: {both_count:,} ({both_count/len(df)*100:.2f}%)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY AND CONCLUSIONS")
print("="*70)

print("\n[TASKS] ANALYSIS PHASES COMPLETED:")
print("  1. [DONE] Introduction - Project scope and objectives")
print("  2. [DONE] Data Source - Dataset origin and composition")
print("  3. [DONE] Data Loading - Successfully loaded from CSV")
print("  4. [DONE] Dataset Structure - Dimensions and column information")
print("  5. [DONE] Data Type Distribution - Column type breakdown")
print("  6. [DONE] Missing Values Analysis - Data quality assessment")
print("  7. [DONE] Duplicate Records - Integrity verification")
print("  8. [DONE] Descriptive Statistics - Numeric variable summaries")
print("  9. [DONE] Delay Analysis - Comprehensive delay pattern analysis")
print("  10. [DONE] Outlier Detection - IQR-based outlier identification")
print("  11. [DONE] Airport Analysis - Geographic distribution")
print("  12. [DONE] Airline Analysis - Carrier representation")

print("\n[OK] Data loading and exploration completed successfully!")
print("\nKey Findings:")
print(f"  • {len(df):,} flight records loaded")
print(f"  • {len(all_airports)} major airport hubs analyzed")
print(f"  • {delayed:,} flights experienced delays")
print(f"  • Data completeness: {completeness:.2f}%")
print(f"  • {len(airline_counts)} different airlines represented")

print("\n[OK] Dataset is prepared for the next preprocessing phase.")
print("="*70)
