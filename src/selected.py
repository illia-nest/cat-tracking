"""Creates selected.txt from selected_copy"""
from pathlib import Path

COPY_FOLDER = Path('data/selected')

paths = [
    p.relative_to(COPY_FOLDER)
    for p in COPY_FOLDER.rglob('*')
    if p.is_file()
]

with open('data/selected.txt', 'w', encoding='utf-8') as f:
    f.writelines(p.as_posix() + '\n' for p in paths)