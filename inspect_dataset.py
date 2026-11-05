from complaint_analyzer.data_utils import read_table

file_path = r"C:\Users\DELL\Downloads\Consumer Complain.xlsx"
df = read_table(file_path)

print("\nCOMPLAINT TYPE DISTRIBUTION")
print("=" * 50)

print(
    df["COMPLAINT_TYPE"]
    .value_counts(dropna=False)
)

print("\nCOMPLAINT SUBTYPE DISTRIBUTION")
print("=" * 50)

print(
    df["COMPLAINT_SUBTYPE"]
    .value_counts(dropna=False)
)

print("\nUNIQUE SUBTYPES:")
print(
    df["COMPLAINT_SUBTYPE"]
    .nunique()
)
print("\nCOMPLETE SUBTYPE DISTRIBUTION")
print("=" * 50)

subtype_counts = df["COMPLAINT_SUBTYPE"].value_counts(dropna=False)

print(subtype_counts.to_string())

print("\nCATEGORY FREQUENCY ANALYSIS")
print("=" * 50)

print("Categories with >= 20 complaints:",
      (subtype_counts >= 20).sum())

print("Categories with 5-19 complaints:",
      ((subtype_counts >= 5) & (subtype_counts < 20)).sum())

print("Categories with fewer than 5 complaints:",
      (subtype_counts < 5).sum())