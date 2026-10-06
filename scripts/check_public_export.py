"""Check the distributable tree and archive members before publishing."""
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {'.tex', '.bib', '.bbl', '.cls', '.bst', '.aux', '.synctex'}
PRIVATE_NAMES = {'agents.md', 'claude.md', 'skill.md', 'core_assessment.md',
                 'core_draft.md', 'gray_theory_note.md', 'gray_theory_assessment.md',
                 'gray_structure.md', 'novelty_matrix.csv', 'decision.json'}
SECRETS = re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
PRIVATE_PATHS = re.compile(rb'C:[/\\]Users[/\\]|E:[/\\]WaveForge|/(?:home|Users)/[^/\s]+/')


def check(name, content):
    path = PurePosixPath(name)
    parts = {p.lower() for p in path.parts}
    if (path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.lower() in PRIVATE_NAMES
            or {'paper', '.codex', '.venv', '.tools', '.external', '.git'} & parts
            or 'prompt' in path.name.lower()):
        raise ValueError(f'non-distributable file: {name}')
    if SECRETS.search(content):
        raise ValueError(f'credential material: {name}')
    if path.suffix.lower() in {'.json', '.jsonl', '.csv', '.md', '.txt', '.yaml', '.yml', '.py'}:
        if PRIVATE_PATHS.search(content):
            raise ValueError(f'private machine path: {name}')


def main():
    names = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard'],
        cwd=ROOT, text=True).splitlines()
    archive_count = 0
    for name in names:
        data = (ROOT/name).read_bytes()
        check(name, data)
        if name.endswith('.zip'):
            with zipfile.ZipFile(ROOT/name) as archive:
                for member in archive.namelist():
                    if member.endswith('/'): continue
                    check(member, archive.read(member))
                    archive_count += 1
    print(f'Public export: {len(names)} files and {archive_count} archive members checked.')


if __name__ == '__main__':
    main()
