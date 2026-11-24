import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD RAW MOTOR INSURANCE DATA
# ============================================================

input_file = "/create-export/create/homes/Sejal.Kothari@sas.com/quick-start-git/Motor_Insurance.csv"

output_file = "/create-export/create/homes/Sejal.Kothari@sas.com/quick-start-git/Motor_Insurance_Prepared.csv"

df = pd.read_csv(input_file)

print("\n========== RAW DATA ==========")
print("Rows and columns:", df.shape)
print(df.head())


# ============================================================
# 2. DATA QUALITY CHECK
# ============================================================

print("\n========== DATA QUALITY CHECKERS ==========")

print("Duplicate Policy IDs:",
      df["Policy_ID"].duplicated().sum())

print("\nMissing values:")
print(df.isnull().sum())


# Remove duplicates
df = df.drop_duplicates(subset=["Policy_ID"])

# Convert date
df["Policy_Start_Date"] = pd.to_datetime(
    df["Policy_Start_Date"],
    errors="coerce"
)


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# Create Mileage Band
df["Mileage_Band"] = pd.cut(
    df["Annual_Mileage_KM"],
    bins=[0, 10000, 20000, float("inf")],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

# Create Customer Age Band
df["Customer_Age_Band"] = pd.cut(
    df["Customer_Age"],
    bins=[0, 29, 50, float("inf")],
    labels=["Young", "Middle", "Mature"]
)

# Create previous claim indicator
df["Has_Previous_Claim"] = (
    df["Previous_Claims_3Y"] > 0
).astype(int)


# ============================================================
# 4. INSURANCE PORTFOLIO SUMMARY
# ============================================================

print("\n========== INSURANCE SUMMARY ==========")

print("Total Policies:", len(df))

print(
    "Policies with Claims:",
    df["Claim_Within_12M"].sum()
)

print(
    "Claim Rate:",
    round(df["Claim_Within_12M"].mean() * 100, 2),
    "%"
)

print(
    "Average Annual Premium: AUD",
    round(df["Annual_Premium_AUD"].mean(), 2)
)

print(
    "Average Claim Amount: AUD",
    round(
        df.loc[
            df["Claim_Within_12M"] == 1,
            "Claim_Amount_AUD"
        ].mean(),
        2
    )
)


# ============================================================
# 5. CLAIM RATE BY VEHICLE TYPE
# ============================================================

vehicle_claims = (
    df.groupby("Vehicle_Type")
      ["Claim_Within_12M"]
      .mean()
      .mul(100)
      .sort_values(ascending=False)
)

print("\n========== CLAIM RATE BY VEHICLE ==========")
print(vehicle_claims.round(2))

vehicle_claims.plot(
    kind="bar",
    title="Claim Rate by Vehicle Type"
)

plt.xlabel("Vehicle Type")
plt.ylabel("Claim Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# 6. CLAIM RATE BY PREVIOUS CLAIMS
# ============================================================

previous_claims = (
    df.groupby("Previous_Claims_3Y")
      ["Claim_Within_12M"]
      .mean()
      .mul(100)
)

print("\n========== CLAIM RATE BY PREVIOUS CLAIMS ==========")
print(previous_claims.round(2))

previous_claims.plot(
    kind="bar",
    title="Claim Rate by Previous Claims"
)

plt.xlabel("Previous Claims in Last 3 Years")
plt.ylabel("Claim Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ============================================================
# 7. CLAIM RATE BY MILEAGE BAND
# ============================================================

mileage_claims = (
    df.groupby(
        "Mileage_Band",
        observed=False
    )["Claim_Within_12M"]
      .mean()
      .mul(100)
)

print("\n========== CLAIM RATE BY MILEAGE ==========")
print(mileage_claims.round(2))

mileage_claims.plot(
    kind="bar",
    title="Claim Rate by Annual Mileage"
)

plt.xlabel("Mileage Band")
plt.ylabel("Claim Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ============================================================
# 8. CLAIM RATE BY CUSTOMER AGE
# ============================================================

age_claims = (
    df.groupby(
        "Customer_Age_Band",
        observed=False
    )["Claim_Within_12M"]
      .mean()
      .mul(100)
)

print("\n========== CLAIM RATE BY CUSTOMER AGE ==========")
print(age_claims.round(2))

age_claims.plot(
    kind="bar",
    title="Claim Rate by Customer Age"
)

plt.xlabel("Customer Age Band")
plt.ylabel("Claim Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ============================================================
# 9. CLAIM AMOUNT DISTRIBUTION
# ============================================================

claims_only = df[
    df["Claim_Amount_AUD"] > 0
]

plt.hist(
    claims_only["Claim_Amount_AUD"],
    bins=20
)

plt.title("Distribution of Claim Amounts")
plt.xlabel("Claim Amount (AUD)")
plt.ylabel("Number of Claims")
plt.tight_layout()
plt.show()


# ============================================================
# 10. SAVE PREPARED DATA FOR MODEL STUDIO
# ============================================================

df.to_csv(
    output_file,
    index=False
)

print("\n========== PREPARED DATA ==========")

print("Rows and columns:", df.shape)

print("\nFinal columns:")
print(df.columns.tolist())

print("\nSample:")
print(df.head(10))

print(
    "\nPrepared dataset saved to:",
    output_file
)