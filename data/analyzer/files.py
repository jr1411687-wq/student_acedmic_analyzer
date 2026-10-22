import re
from pathlib import Path

from analyzer.catalog import _label_from_filename
from analyzer.constants import DATA_DIR

def _iter_data_files(pattern):
    """Find data files in data/ and nested folders like 'all results/'."""
    files = []
    # Direct children of data/
    files.extend(DATA_DIR.glob(pattern))
    # Nested folders (e.g. data/all results/*.csv)
    files.extend(DATA_DIR.glob(f"*/{pattern}"))
    files.extend(DATA_DIR.glob(f"**/{pattern}"))
    # Unique by resolved path, skip junk dirs
    unique = {}
    for f in files:
        if not f.is_file():
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in f.parts):
            continue
        unique[str(f.resolve())] = f
    return sorted(unique.values(), key=lambda p: p.name.lower())


def is_section_list_file(path):
    """True for section-allocation sheets (not exam result files)."""
    name = Path(path).name.lower().replace("_", " ").replace("-", " ")
    stem = Path(path).stem.lower()
    if "section list" in name or stem.startswith("section_list") or stem.startswith("sectionlist"):
        return True
    if re.search(r"\bsections?\b", name) and re.search(r"\b(list|allot|alloc|roll)\b", name):
        return True
    return False


def discover_csv_files():
    """Auto-discover CSV files in data/ and subfolders (e.g. all results/)."""
    datasets = {}
    for f in _iter_data_files("*.csv"):
        if is_section_list_file(f):
            continue
        label = _label_from_filename(f.name)
        # Keep labels unique if same name appears twice
        if label in datasets:
            parent = f.parent.name
            label = f"{label} [{parent}]"
        datasets[label] = str(f)
    return datasets


def discover_pdf_files():
    """Auto-discover PDF files in data/ and subfolders (e.g. all results/)."""
    datasets = {}
    for f in _iter_data_files("*.pdf"):
        if is_section_list_file(f):
            continue
        label = _label_from_filename(f.name) + " (PDF)"
        if label in datasets:
            parent = f.parent.name
            label = f"{_label_from_filename(f.name)} [{parent}] (PDF)"
        datasets[label] = str(f)
    return datasets


def discover_data_files():
    """Discover CSV and PDF result datasets in the data folder tree."""
    datasets = {}
    datasets.update(discover_csv_files())
    datasets.update(discover_pdf_files())
    return datasets


def discover_section_list_files():
    """Discover section-list CSV/PDF files (Roll → Section maps)."""
    files = []
    for pattern in ("*.csv", "*.pdf", "*.xlsx", "*.xls"):
        for f in _iter_data_files(pattern):
            if is_section_list_file(f):
                files.append(f)
    # Prefer CSVs first (cleaner than OCR PDFs), then by name
    files = sorted(files, key=lambda p: (p.suffix.lower() != ".csv", p.name.lower()))
    return files
