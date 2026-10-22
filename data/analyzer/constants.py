import re
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent
FRAME_CACHE_DIR = DATA_DIR / ".cache_frames"
META_CACHE_DIR = DATA_DIR / ".cache_meta"
GRADE_PATTERN = re.compile(
    r'^(O|A\+|A|B\+|B|C|D|E|F|S|P|Ab|AB|NA|NE|F\*|[A-F][+-]?)(\s*\([^)]+\))?$',
    re.IGNORECASE
)
ROLL_PATTERN = re.compile(r'^\s*(\d+)\s+(\d{7})\s+(.*)$')
SGPA_CREDITS_PATTERN = re.compile(r'([\d.]+)\s+(\d+)\s*$')

METADATA_ALIASES = {
    'name': [
        'name', 'student name', 'name of candidate', 'candidate name',
        'student_name', 'candidate_name', 'studentname', 'candidatename'
    ],
    'roll_no': [
        'roll_no', 'roll no', 'roll no.', 'roll number', 'roll',
        'university roll no', 'university_roll_no', 'university roll no.'
    ],
    'sgpa': ['sgpa', 'sgpa**', 'sgpa*', 'gpa', 'cgpa'],
    'credits': [
        'credits_earned', 'credits earned', 'credits', 'credit',
        'credits (earned)', 'credits earned'
    ],
    'gender': ['gender', 'sex'],
    'father_name': ['father_name', "father's name", 'father name', 'father'],
}

BRANCH_ALIASES = {
    'CSE': ['cse', 'computer science', 'computer-science'],
    'IT': ['it', 'information technology', 'information-technology'],
    'ECE': ['ece', 'electronics', 'electronics & communication', 'electronics and communication'],
    'EE': ['ee', 'electrical'],
    'ME': ['me', 'mechanical'],
    'Civil': ['civil', 'ce'],
}

SUBJECT_NAME_PATTERNS = [
    r'^[A-Z]{2,5}[-_]?\d{2,3}$',       # BSC_101, BSEC101, ESC-101
    r'^[A-Z]{2,5}[-_]\d{3}$',
    r'^[A-Z]{2,5}\s*\(\d+\)',           # BSEC101 (96371)
    r'^PRIT-\d+',
    r'^TR-\d+',
    r'^CEC\d+',
    r'^BSEC\d+',
    r'^SMEC\d+',
    r'^TREC\d+',
    r'^HSMC\d+',
    r'^MC\d+',
    r'elective',
    r'open elective',
    r'^[A-Z]{3,5}_\d{3}$',
]

