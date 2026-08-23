import pdfplumber
import pandas as pd
import re

PDF = "data/ece_original.pdf"
CSV = "data/ece_sem3.csv"

# Positions of the 7 subject-grade columns
GRADE_COLS = [4, 5, 6, 8, 9, 10, 12]

records = []

with pdfplumber.open(PDF) as pdf:

    for page_no, page in enumerate(pdf.pages, start=1):

        tables = page.extract_tables()

        for table in tables:

            for row in table:

                if not row or len(row) < 15:
                    continue

                # Clean cells
                row = [
                    cell.strip() if cell else ""
                    for cell in row
                ]

                # Roll number must be 7 digits
                if not re.fullmatch(r"\d{7}", row[1]):
                    continue

                roll_no = row[1]

                # Name + Father name are in the same cell
                person = row[2]

                parts = [
                    x.strip()
                    for x in person.split("\n")
                    if x.strip()
                ]

                if len(parts) >= 1:
                    name = parts[0]
                else:
                    name = ""

                if len(parts) >= 2:
                    father_name = parts[1]
                else:
                    father_name = ""

                # Extract 7 grades
                grades = [row[i] for i in GRADE_COLS]

                # SGPA
                sgpa = row[13]

                # Credits
                credits = row[14]

                records.append({
                    "Roll_No": roll_no,
                    "Name": name,
                    "Father_Name": father_name,
                    "BSC_101": grades[0],
                    "ESC_101": grades[1],
                    "BSC_102": grades[2],
                    "ESC_102": grades[3],
                    "BSC_103": grades[4],
                    "ESC_103": grades[5],
                    "HSMC_103": grades[6],
                    "SGPA": sgpa,
                    "Credits_Earned": credits,
                    "Page": page_no
                })

# Create DataFrame
df = pd.DataFrame(records)

# Remove duplicate roll numbers
df = df.drop_duplicates(
    subset=["Roll_No"],
    keep="first"
)

# Convert numeric fields
df["SGPA"] = pd.to_numeric(
    df["SGPA"],
    errors="coerce"
)

df["Credits_Earned"] = pd.to_numeric(
    df["Credits_Earned"],
    errors="coerce"
)

# Remove helper page column
df = df.drop(columns=["Page"])

# Save
df.to_csv(CSV, index=False)

print("\n==============================")
print("CSV CREATED SUCCESSFULLY")
print("==============================")
print("Students found:", len(df))
print("File:", CSV)

# Check Jatin Singh
jatin = df[
    df["Roll_No"].astype(str) == "2303140"
]

print("\nJATIN SINGH CHECK:")
print(jatin.to_string(index=False))

print("\nFirst 5 students:")
print(df.head().to_string(index=False))
