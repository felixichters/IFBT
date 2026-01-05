#!/usr/bin/env python3
# This script processes multiple .tar.gz archives outputted from the CompilePipeline.py script,
# in the current directory. It extracts executable binaries from each archive, renames them
# and consolidates them into a single directory.

import tarfile
import os
import shutil
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm
import argparse

# Configuration
INPUT_DIR = Path("COMPILED")
OUTPUT_DIR = Path("executables")
FINAL_ARCHIVE = Path("all_executables.tar.gz")

OUTPUT_DIR.mkdir(exist_ok=True)

def is_executable_candidate(tarinfo: tarfile.TarInfo) -> bool:
    """Determine if a TarInfo member is an executable binary."""
    if not tarinfo.isfile():
        return False

    name = tarinfo.name

    if name.endswith(".o") or Path(name).is_file():
        return False
    # Heuristic: consider files in paths starting with "executable" as candidates
    return name.split("/")[-1].startswith("executable")

def main(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

    archives = [x for x in input_dir.glob("*.tar.gz")]
    for archive_path in tqdm(archives, desc="Processing archives"):
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                # Collect executables per top-level folder
                execs_per_dir = defaultdict(list)
        
                for member in tar.getmembers():
                    if not is_executable_candidate(member):
                        continue
                        
                    path = Path(member.name)
                    if len(path.parts) < 2:
                        continue
        
                    top_dir = path.parts[0]
                    execs_per_dir[top_dir].append(member)
        
                # Extract and rename
                for top_dir, members in execs_per_dir.items():
                    single_exec = len(members) == 1
        
                    for member in members:
                        path = Path(member.name)
                        exe_name = path.name
        
                        if single_exec:
                            new_name = top_dir
                        else:
                            new_name = f"{top_dir}_{exe_name}"
        
                        output_path = OUTPUT_DIR / new_name
        
                        fileobj = tar.extractfile(member)
                        if fileobj is None:
                            continue
        
                        with open(output_path, "wb") as f:
                            shutil.copyfileobj(fileobj, f)
        
                        os.chmod(output_path, member.mode)
        except tarfile.ReadError:
            # Empty or invalid archive
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge and extract executable binaries from multiple .tar.gz archives.")

    parser.add_argument('--dest', type=str, default=OUTPUT_DIR, help='Output directory path that will the final executables (default: executables).')
    parser.add_argument('--source', type=str, default=INPUT_DIR, help='Path to directory that contains the .tar.gz archives to process (default: COMPILED).')
    # Parsing the arguments
    args = parser.parse_args()
    main(input_dir=Path(args.source), output_dir=Path(args.dest))
