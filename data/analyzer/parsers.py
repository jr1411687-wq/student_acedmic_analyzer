import pandas as pd
import os
import re
from pathlib import Path

from analyzer.constants import GRADE_PATTERN, METADATA_ALIASES, ROLL_PATTERN, SGPA_CREDITS_PATTERN, SUBJECT_NAME_PATTERNS

def _read_text_sample(source, max_lines=30):
    """Read first lines from a file path, bytes buffer, or file-like object."""
    if isinstance(source, (str, Path)):
        with open(source, 'r', encoding='utf-8', errors='ignore') as f:
            return [line.rstrip('\n') for line in f.readlines()[:max_lines]]
    if isinstance(source, bytes):
        text = source.decode('utf-8', errors='ignore')
        return text.splitlines()[:max_lines]
    pos = source.tell() if hasattr(source, 'tell') else None
    raw = source.read()
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8', errors='ignore')
    if pos is not None and hasattr(source, 'seek'):
        source.seek(pos)
    return raw.splitlines()[:max_lines]


def detect_csv_format(source):
    """Detect whether CSV is standard tabular or college result format."""
    sample = _read_text_sample(source)
    if not sample:
        return 'standard'

    first_line = sample[0].strip()

    # Standard CSV starts with a comma-separated header row
    if ',' in first_line and first_line.count(',') >= 3:
        header_lower = first_line.lower()
        if not header_lower.startswith('"guru') and 'guru nanak' not in header_lower:
            if any(kw in header_lower for kw in ('roll', 'sgpa', 'name', 'credit')):
                return 'standard'

    text = '\n'.join(sample).lower()
    if 'guru nanak dev engineering college' in text:
        return 'college_result'
    if 'sgpa**' in text and 'sub code' in text:
        return 'college_result'
    if 'notification no.' in text and 'roll no.' in text:
        return 'college_result'

    return 'standard'


def _clean_line(line):
    return line.strip().strip('"').strip()


def _extract_subject_headers(line):
    """Extract subject column names from a college result header row."""
    parts = re.split(r'\s{2,}', line.strip())
    subjects = []
    for part in parts:
        part = part.strip()
        if not part or part.upper().startswith('SGPA'):
            continue
        subjects.append(part)
    return subjects


def parse_college_result_csv(source):
    """Parse GNDEC-style multi-row college result CSV files."""
    if isinstance(source, (str, Path)):
        with open(source, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    else:
        pos = source.tell() if hasattr(source, 'tell') else None
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8', errors='ignore')
        lines = raw.splitlines()
        if pos is not None and hasattr(source, 'seek'):
            source.seek(pos)

    records = []
    subject_names = []
    default_subjects = [
        'Subject_1', 'Subject_2', 'Subject_3', 'Subject_4',
        'Subject_5', 'Subject_6', 'Subject_7'
    ]

    i = 0
    while i < len(lines):
        line = _clean_line(lines[i])

        if 'Roll No.' in line:
            for offset in (1, 2, 3):
                if i - offset < 0:
                    continue
                header_line = _clean_line(lines[i - offset])
                if any(kw in header_line for kw in ('PRIT', 'Elective', 'BSC', 'ESC', 'TR-', 'HSMC', 'Sub Code')):
                    parsed_subjects = _extract_subject_headers(header_line)
                    if parsed_subjects:
                        subject_names = parsed_subjects
                        break
                if 'SGPA' in header_line.upper() and not header_line.upper().startswith('SR'):
                    parsed_subjects = _extract_subject_headers(header_line)
                    if parsed_subjects:
                        subject_names = parsed_subjects
                        break

        match = ROLL_PATTERN.match(line)
        if match:
            roll_no = match.group(2)
            rest = match.group(3)

            name = ''
            if i > 0:
                prev = _clean_line(lines[i - 1])
                if prev and not ROLL_PATTERN.match(prev) and 'Credit' not in prev:
                    if not any(kw in prev for kw in ('Roll No', 'SGPA', 'Sub Code', 'Sr.')):
                        name = prev.rstrip('/').strip()

            father_name = ''
            if i + 1 < len(lines):
                nxt = _clean_line(lines[i + 1])
                if nxt and not ROLL_PATTERN.match(nxt):
                    if not any(kw in nxt for kw in ('**', 'Page:', 'GURU NANAK', 'Course:')):
                        father_name = nxt

            end_match = SGPA_CREDITS_PATTERN.search(rest)
            if end_match:
                sgpa = end_match.group(1)
                credits = end_match.group(2)
                grades_part = rest[:end_match.start()].strip()
                grades = [g.strip() for g in re.split(r'\s{2,}', grades_part) if g.strip()]
            else:
                sgpa, credits, grades = None, None, []

            subjects = subject_names if subject_names else default_subjects
            record = {
                'Roll_No': roll_no,
                'Name': name,
                'Father_Name': father_name,
                'SGPA': sgpa,
                'Credits_Earned': credits,
            }
            for idx, grade in enumerate(grades):
                col_name = subjects[idx] if idx < len(subjects) else f'Subject_{idx + 1}'
                record[_safe_column_name(col_name)] = grade
            records.append(record)

        i += 1

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset=['Roll_No'], keep='first')
    return df


def _safe_column_name(name):
    """Convert subject header text to a safe column name."""
    safe = re.sub(r'[^\w\s-]', '', str(name)).strip()
    safe = re.sub(r'\s+', '_', safe)
    return safe or 'Subject'


def _find_column(columns, aliases, exclude=None):
    """Find a column matching any alias (case-insensitive)."""
    exclude_lower = {str(e).strip().lower() for e in (exclude or [])}
    col_map = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias in col_map and alias not in exclude_lower:
            return col_map[alias]
    for col in columns:
        col_lower = str(col).strip().lower()
        if col_lower in exclude_lower:
            continue
        for alias in aliases:
            if alias == 'name' and 'father' in col_lower and '/' not in col:
                continue
            if alias in col_lower or col_lower in alias:
                return col
    return None


def detect_metadata_columns(df):
    """Auto-detect Name, Roll, SGPA, Credits, Gender columns."""
    name_col = _find_column(df.columns, METADATA_ALIASES['name'])
    father_col = _find_column(df.columns, METADATA_ALIASES['father_name'])

    # Combined "Name/Father's Name" column should only map to Name
    if name_col and father_col and name_col == father_col:
        father_col = None

    return {
        'name': name_col,
        'roll_no': _find_column(df.columns, METADATA_ALIASES['roll_no']),
        'sgpa': _find_column(df.columns, METADATA_ALIASES['sgpa']),
        'credits': _find_column(df.columns, METADATA_ALIASES['credits']),
        'gender': _find_column(df.columns, METADATA_ALIASES['gender']),
        'father_name': father_col,
    }


def _is_grade_value(val):
    """Check if a value looks like an academic grade."""
    if pd.isna(val) or val == '':
        return False
    val_str = str(val).strip()
    if GRADE_PATTERN.match(val_str):
        return True
    if re.match(r'^[A-F][+-]?\s*\([^)]+\)$', val_str, re.IGNORECASE):
        return True
    return False


def _is_subject_column_name(col_name):
    """Check if column name looks like a subject code."""
    col_str = str(col_name).strip()
    for pattern in SUBJECT_NAME_PATTERNS:
        if re.search(pattern, col_str, re.IGNORECASE):
            return True
    return False


def detect_subject_columns(df, metadata_cols=None):
    """Auto-detect subject/grade columns from any CSV structure."""
    if metadata_cols is None:
        metadata_cols = detect_metadata_columns(df)

    excluded = {
        metadata_cols.get(k) for k in metadata_cols if metadata_cols.get(k)
    }
    excluded.update({'Page', 'Pass_Status', 'Performance_Category', 'SGPA_Range'})

    subject_cols = []
    for col in df.columns:
        if col in excluded or col in subject_cols:
            continue

        if _is_subject_column_name(col):
            subject_cols.append(col)
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        grade_ratio = series.apply(_is_grade_value).mean()
        if grade_ratio >= 0.5:
            subject_cols.append(col)

    return subject_cols


def _split_combined_name_field(df):
    """Split 'Name/Father Name' combined columns into separate fields."""
    if 'Name' not in df.columns:
        return df

    combined = df['Name'].astype(str)
    if not combined.str.contains('/', na=False).any():
        return df

    if 'Father_Name' not in df.columns:
        split_names = combined.str.split('/', n=1, expand=True)
        df['Name'] = split_names[0].str.strip()
        if split_names.shape[1] > 1:
            df['Father_Name'] = split_names[1].str.strip()
    else:
        df['Name'] = combined.str.split('/', n=1).str[0].str.strip()

    return df


def normalize_dataframe(df):
    """Normalize detected columns to standard names for analysis."""
    meta = detect_metadata_columns(df)
    rename_map = {}
    if meta['name'] and meta['name'] != 'Name':
        rename_map[meta['name']] = 'Name'
    if meta['roll_no'] and meta['roll_no'] != 'Roll_No':
        rename_map[meta['roll_no']] = 'Roll_No'
    if meta['sgpa'] and meta['sgpa'] != 'SGPA':
        rename_map[meta['sgpa']] = 'SGPA'
    if meta['credits'] and meta['credits'] != 'Credits_Earned':
        rename_map[meta['credits']] = 'Credits_Earned'
    if meta['gender'] and meta['gender'] != 'Gender':
        rename_map[meta['gender']] = 'Gender'
    if meta['father_name'] and meta['father_name'] != 'Father_Name':
        rename_map[meta['father_name']] = 'Father_Name'

    df = df.rename(columns=rename_map)
    df = _split_combined_name_field(df)

    if 'Roll_No' in df.columns:
        df['Roll_No'] = df['Roll_No'].astype(str).str.strip()

    return df, detect_subject_columns(df, meta)


def load_csv_smart(source, filename=''):
    """Load any CSV with auto format detection and column normalization."""
    fmt = detect_csv_format(source)

    if fmt == 'college_result':
        df = parse_college_result_csv(source)
    else:
        if isinstance(source, (str, Path)):
            df = pd.read_csv(source)
        else:
            pos = source.tell() if hasattr(source, 'tell') else None
            df = pd.read_csv(source)
            if pos is not None and hasattr(source, 'seek'):
                source.seek(pos)

    if df.empty:
        return df, []

    df, subject_cols = normalize_dataframe(df)
    return df, subject_cols


# =========================================================
# PDF RESULT PARSING
# =========================================================

def _open_pdf_source(source):
    """Return (pdfplumber PDF, tempfile_path_or_None) for path/bytes/file-like."""
    import pdfplumber
    import tempfile

    if isinstance(source, (str, Path)):
        return pdfplumber.open(str(source)), None

    if isinstance(source, bytes):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(source)
        tmp.close()
        return pdfplumber.open(tmp.name), tmp.name

    # Streamlit UploadedFile / file-like
    pos = source.tell() if hasattr(source, "tell") else None
    raw = source.read()
    if pos is not None and hasattr(source, "seek"):
        source.seek(pos)
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="ignore")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(raw)
    tmp.close()
    return pdfplumber.open(tmp.name), tmp.name


def _is_roll_cell(val):
    return bool(re.fullmatch(r"\d{7}", str(val or "").strip()))


def _looks_like_grade_cell(val):
    if val is None or val == "":
        return False
    return _is_grade_value(str(val).strip().replace("\n", " "))


def _split_person_cell(person):
    parts = [x.strip() for x in str(person or "").replace("/", "\n").split("\n") if x.strip()]
    name = parts[0] if parts else ""
    father = parts[1] if len(parts) > 1 else ""
    return name, father


def _extract_subjects_from_header_row(row):
    """Return list of (col_index, subject_name) from a PDF header row."""
    subjects = []
    for i, cell in enumerate(row):
        if not cell:
            continue
        text = str(cell).strip().replace("\n", " ")
        low = text.lower()
        if not text or low.startswith("sgpa") or "earned" in low:
            continue
        if low in ("sr.", "sr. no.", "no.", "university roll no.", "name of candidate/ father's name"):
            continue
        if re.search(
            r"[A-Z]{2,6}[-_]?\d{2,3}|Elective|PRIT|TR-|CEC|BSC|ESC|HSMC|BSEC|SMEC|TREC|ESEC|MCEC|MPD|MC\d|Sub Code",
            text,
            re.I,
        ):
            # Prefer short subject code from header text
            code = re.search(r"([A-Z]{2,6}[-_]?\d{2,3})", text, re.I)
            label = code.group(1).upper() if code else _safe_column_name(text)
            subjects.append((i, label))
    return subjects


def _find_sgpa_credits_indices(row):
    """Locate SGPA and Credits column indices from a header row."""
    sgpa_idx, credits_idx = None, None
    for i, cell in enumerate(row):
        if not cell:
            continue
        text = str(cell).strip().replace("\n", " ").lower()
        if "sgpa" in text:
            sgpa_idx = i
        elif "earned" in text or (text.startswith("credit") and "sub code" not in text):
            credits_idx = i
    return sgpa_idx, credits_idx


def _parse_sgpa_credits_values(sgpa_raw, credits_raw, trailing_numbers=None):
    """
    Correctly assign SGPA vs Credits.
    - SGPA is blank often; never treat standalone credits as SGPA.
    - SGPA is typically 0-10 (often decimal); credits are usually integers.
    """
    sgpa, credits = None, None

    def _clean_num(val):
        if val is None:
            return None
        s = str(val).strip().replace("\n", " ")
        if not s or not re.fullmatch(r"\d+(\.\d+)?", s):
            return None
        return s

    sgpa = _clean_num(sgpa_raw)
    credits = _clean_num(credits_raw)

    # If header cells were empty, use trailing numbers carefully
    if trailing_numbers:
        nums = [n for n in trailing_numbers if _clean_num(n)]
        if sgpa is None and credits is None:
            if len(nums) >= 2:
                a, b = nums[-2], nums[-1]
                try:
                    fa, fb = float(a), float(b)
                    # Prefer decimal / <=10 as SGPA, larger/int as credits
                    if fa <= 10 and fb >= fa:
                        sgpa, credits = a, b
                    elif fb <= 10 and fa > 10:
                        sgpa, credits = b, a
                    else:
                        sgpa, credits = a, b
                except ValueError:
                    sgpa, credits = a, b
            elif len(nums) == 1:
                n = nums[0]
                try:
                    fn = float(n)
                    # Single trailing value: credits if integer-like, else SGPA
                    if "." not in n:
                        credits = n
                    elif fn <= 10:
                        sgpa = n
                    else:
                        credits = n
                except ValueError:
                    credits = n
        elif sgpa is None and credits is None:
            pass
        elif sgpa is None and credits is not None:
            pass  # blank SGPA is valid
        elif credits is None and sgpa is not None:
            # Maybe mis-read: integer-only "SGPA" with no credits may actually be credits
            if "." not in str(sgpa):
                try:
                    if float(sgpa) > 10 or float(sgpa).is_integer():
                        # Keep as SGPA only if clearly grade-point-ish (<=10) AND we expect SGPA present
                        # If value is whole number <=10 with no credits col filled, leave as SGPA
                        # (ALPANA case: 7 and 26 both filled — handled above)
                        pass
                except ValueError:
                    pass

    # Final sanity: swap if clearly reversed
    if sgpa is not None and credits is not None:
        try:
            fs, fc = float(sgpa), float(credits)
            if fs > 10 and fc <= 10:
                sgpa, credits = credits, sgpa
        except ValueError:
            pass

    return sgpa, credits


def parse_result_pdf(source):
    """
    Parse GNDEC-style result PDFs into a student DataFrame.
    Uses header column positions for accurate SGPA / Credits.
    """
    pdf, tmp_path = _open_pdf_source(source)
    records = []
    subject_cols = []  # list of (idx, name)
    sgpa_idx, credits_idx = None, None

    try:
        with pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables() or []
                for table in tables:
                    if not table:
                        continue

                    # Skip grade-legend tables
                    first_cell = (table[0][0] or "").strip().lower() if table[0] else ""
                    if first_cell in ("grade", "qualitative meaning", "grade point"):
                        continue

                    for row in table:
                        if not row:
                            continue
                        cleaned = [
                            (c.strip() if isinstance(c, str) else ("" if c is None else str(c).strip()))
                            for c in row
                        ]
                        joined = " ".join(cleaned).upper()

                        # Header row: refresh subject / SGPA / credits indices
                        if "SGPA" in joined and (
                            "SUB CODE" in joined or "ROLL" in joined or "CANDIDATE" in joined
                        ):
                            found_subjects = _extract_subjects_from_header_row(cleaned)
                            s_idx, c_idx = _find_sgpa_credits_indices(cleaned)
                            if found_subjects:
                                subject_cols = found_subjects
                            if s_idx is not None:
                                sgpa_idx = s_idx
                            if c_idx is not None:
                                credits_idx = c_idx
                            continue

                        # Student row must have a 7-digit roll
                        roll_idx = None
                        for i, cell in enumerate(cleaned):
                            if _is_roll_cell(cell):
                                roll_idx = i
                                break
                        if roll_idx is None:
                            continue

                        roll_no = cleaned[roll_idx]
                        person = cleaned[roll_idx + 1] if roll_idx + 1 < len(cleaned) else ""
                        name, father_name = _split_person_cell(person)

                        # Grades from known subject columns, else scan middle cells
                        grade_pairs = []
                        if subject_cols:
                            for col_i, col_name in subject_cols:
                                if col_i < len(cleaned):
                                    g = cleaned[col_i].replace("\n", " ").strip()
                                    if g:
                                        grade_pairs.append((col_name, g))
                        else:
                            for i, cell in enumerate(cleaned):
                                if i <= roll_idx + 1:
                                    continue
                                if sgpa_idx is not None and i == sgpa_idx:
                                    continue
                                if credits_idx is not None and i == credits_idx:
                                    continue
                                g = cell.replace("\n", " ").strip()
                                if _looks_like_grade_cell(g):
                                    grade_pairs.append((f"Subject_{len(grade_pairs)+1}", g))

                        # SGPA / Credits from fixed header indices
                        sgpa_raw = cleaned[sgpa_idx] if sgpa_idx is not None and sgpa_idx < len(cleaned) else ""
                        credits_raw = cleaned[credits_idx] if credits_idx is not None and credits_idx < len(cleaned) else ""

                        trailing = []
                        if (not sgpa_raw or not credits_raw) and cleaned:
                            # collect trailing numeric cells as fallback only
                            for cell in cleaned[::-1]:
                                cell_s = cell.replace("\n", " ").strip()
                                if re.fullmatch(r"\d+(\.\d+)?", cell_s):
                                    trailing.append(cell_s)
                                elif cell_s:
                                    break
                            trailing = list(reversed(trailing))

                        sgpa, credits = _parse_sgpa_credits_values(
                            sgpa_raw, credits_raw, trailing_numbers=trailing
                        )

                        if not grade_pairs and sgpa is None and credits is None:
                            continue

                        record = {
                            "Roll_No": roll_no,
                            "Name": name,
                            "Father_Name": father_name,
                            "SGPA": sgpa,
                            "Credits_Earned": credits,
                            "Page": page_no,
                        }
                        for col_name, grade in grade_pairs:
                            record[col_name] = grade
                        records.append(record)

            # Text fallback if tables produced nothing
            if not records:
                for page_no, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    lines = text.splitlines()
                    for i, line in enumerate(lines):
                        m2 = re.search(r"\b(\d{7})\b", line)
                        if not m2:
                            continue
                        roll_no = m2.group(1)
                        rest = line[m2.end():].strip()

                        name = ""
                        if i > 0:
                            prev = lines[i - 1].strip()
                            if prev and not re.search(r"\d{7}", prev) and "Credit" not in prev:
                                name = prev.rstrip("/").strip()

                        # Prefer: ... [SGPA] CREDITS at end; credits always last integer
                        nums = re.findall(r"\d+(?:\.\d+)?", rest)
                        grades = [
                            g for g in re.split(r"\s+", rest)
                            if g and _looks_like_grade_cell(g)
                        ]
                        sgpa, credits = None, None
                        if len(nums) >= 2:
                            # last is credits, previous may be SGPA if <= 10
                            credits = nums[-1]
                            maybe_sgpa = nums[-2]
                            try:
                                if float(maybe_sgpa) <= 10:
                                    sgpa = maybe_sgpa
                                else:
                                    # both look like credits-ish; keep only last as credits
                                    credits = nums[-1]
                            except ValueError:
                                sgpa = maybe_sgpa
                        elif len(nums) == 1:
                            # Single number at end of result line is Credits (SGPA often blank)
                            credits = nums[0]

                        if not name and not grades and sgpa is None and credits is None:
                            continue

                        record = {
                            "Roll_No": roll_no,
                            "Name": name,
                            "SGPA": sgpa,
                            "Credits_Earned": credits,
                            "Page": page_no,
                        }
                        for idx, grade in enumerate(grades):
                            record[f"Subject_{idx + 1}"] = grade
                        records.append(record)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["Roll_No"], keep="first")
    if "Page" in df.columns:
        df = df.drop(columns=["Page"])
    return df


def load_pdf_smart(source, filename=''):
    """Parse a result PDF and normalize columns for analysis."""
    df = parse_result_pdf(source)
    if df.empty:
        return df, []
    df, subject_cols = normalize_dataframe(df)
    return df, subject_cols


def load_dataset_smart(source, filename=''):
    """Load CSV or PDF and return (dataframe, subject_columns)."""
    name = (filename or "").lower()
    if not name:
        if isinstance(source, (str, Path)):
            name = str(source).lower()
        elif hasattr(source, "name"):
            name = str(source.name).lower()

    if name.endswith(".pdf"):
        return load_pdf_smart(source, filename)
    return load_csv_smart(source, filename)
