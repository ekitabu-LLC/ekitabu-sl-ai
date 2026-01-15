"""
Find Duplicate Videos by MD5 Hash
Scans the dataset directory and generates a report of duplicate files.

Usage:
    python find_duplicates.py [directory]
    
Example:
    python find_duplicates.py dataset
    python find_duplicates.py validation-data
"""

import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime


def compute_md5(filepath: str, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def find_duplicates(directory: str) -> dict:
    """
    Find all duplicate files in a directory.
    Returns a dict mapping MD5 hash to list of file paths.
    """
    hash_to_files = defaultdict(list)
    
    # Supported video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
    
    # Also check numpy files if this is a processed dataset
    data_extensions = {'.npy'}
    
    all_extensions = video_extensions | data_extensions
    
    total_files = 0
    processed = 0
    
    # First pass: count files
    for root, _, files in os.walk(directory):
        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext in all_extensions:
                total_files += 1
    
    print(f"Found {total_files} files to check...")
    
    # Second pass: compute hashes
    for root, _, files in os.walk(directory):
        for filename in files:
            ext = Path(filename).suffix.lower()
            if ext not in all_extensions:
                continue
                
            filepath = os.path.join(root, filename)
            try:
                file_hash = compute_md5(filepath)
                hash_to_files[file_hash].append(filepath)
                processed += 1
                
                if processed % 100 == 0:
                    print(f"  Processed {processed}/{total_files} files...")
                    
            except Exception as e:
                print(f"  Error reading {filepath}: {e}")
    
    # Filter to only duplicates
    duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
    
    return duplicates


def generate_report(duplicates: dict, output_path: str, directory: str):
    """Generate a detailed duplicate report."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DUPLICATE FILES REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Directory: {os.path.abspath(directory)}\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 70 + "\n\n")
        
        if not duplicates:
            f.write("No duplicates found!\n")
            print("\n✓ No duplicates found!")
            return
        
        # Summary
        total_duplicate_sets = len(duplicates)
        total_duplicate_files = sum(len(files) for files in duplicates.values())
        wasted_files = total_duplicate_files - total_duplicate_sets  # Extra copies
        
        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Duplicate sets found: {total_duplicate_sets}\n")
        f.write(f"Total duplicate files: {total_duplicate_files}\n")
        f.write(f"Files that could be removed: {wasted_files}\n")
        
        # Calculate wasted space
        total_wasted_bytes = 0
        for md5_hash, files in duplicates.items():
            if files:
                file_size = os.path.getsize(files[0])
                total_wasted_bytes += file_size * (len(files) - 1)
        
        if total_wasted_bytes > 1024 * 1024:
            f.write(f"Wasted space: {total_wasted_bytes / (1024*1024):.2f} MB\n")
        else:
            f.write(f"Wasted space: {total_wasted_bytes / 1024:.2f} KB\n")
        
        f.write("\n")
        
        # Count duplicates per class
        class_dup_count = defaultdict(int)
        for md5_hash, files in duplicates.items():
            for filepath in files:
                parts = Path(filepath).parts
                # Try to find the sign class name (usually the parent folder of the file)
                # Handle paths like dataset/lstm_processed/train/125_001.npy
                filename = Path(filepath).stem
                # Extract class from filename (e.g., "125" from "125_001")
                class_name = filename.rsplit('_', 1)[0] if '_' in filename else parts[-2]
                class_dup_count[class_name] += 1
        
        f.write("DUPLICATES PER CLASS\n")
        f.write("-" * 70 + "\n")
        # Sort by count descending
        sorted_classes = sorted(class_dup_count.items(), key=lambda x: -x[1])
        for class_name, count in sorted_classes:
            f.write(f"  {class_name:15s}: {count} duplicate files\n")
        f.write("\n")
        
        # Print to console too
        print("\nDuplicates per class:")
        for class_name, count in sorted_classes[:10]:  # Top 10
            print(f"  {class_name:15s}: {count}")
        if len(sorted_classes) > 10:
            print(f"  ... and {len(sorted_classes) - 10} more classes")
        
        # Group by class/directory
        class_duplicates = defaultdict(list)
        cross_class_duplicates = []
        
        for md5_hash, files in duplicates.items():
            # Get class names from paths
            classes = set()
            for filepath in files:
                parts = Path(filepath).parts
                # Find the class name (usually parent directory of file)
                if len(parts) >= 2:
                    classes.add(parts[-2])
            
            if len(classes) == 1:
                class_duplicates[list(classes)[0]].append((md5_hash, files))
            else:
                cross_class_duplicates.append((md5_hash, files, classes))
        
        # Cross-class duplicates (more concerning)
        if cross_class_duplicates:
            f.write("=" * 70 + "\n")
            f.write("⚠️  CROSS-CLASS DUPLICATES (Same file in different classes!)\n")
            f.write("=" * 70 + "\n")
            f.write("These are the most concerning - same video labeled as different signs.\n\n")
            
            for md5_hash, files, classes in cross_class_duplicates:
                file_size = os.path.getsize(files[0])
                f.write(f"MD5: {md5_hash}\n")
                f.write(f"Size: {file_size:,} bytes\n")
                f.write(f"Classes involved: {', '.join(sorted(classes))}\n")
                f.write("Files:\n")
                for filepath in sorted(files):
                    f.write(f"  - {filepath}\n")
                f.write("\n")
        
        # Within-class duplicates
        if class_duplicates:
            f.write("=" * 70 + "\n")
            f.write("WITHIN-CLASS DUPLICATES (Same file copied within a class)\n")
            f.write("=" * 70 + "\n\n")
            
            for class_name, dup_list in sorted(class_duplicates.items()):
                f.write(f"Class: {class_name} ({len(dup_list)} duplicate sets)\n")
                f.write("-" * 40 + "\n")
                
                for md5_hash, files in dup_list:
                    file_size = os.path.getsize(files[0])
                    f.write(f"  MD5: {md5_hash[:16]}... ({file_size:,} bytes)\n")
                    for filepath in sorted(files):
                        f.write(f"    - {os.path.basename(filepath)}\n")
                f.write("\n")
        
        # Generate removal script
        f.write("=" * 70 + "\n")
        f.write("REMOVAL COMMANDS (keep first, remove rest)\n")
        f.write("=" * 70 + "\n")
        f.write("# Review carefully before running!\n\n")
        
        for md5_hash, files in duplicates.items():
            sorted_files = sorted(files)
            f.write(f"# Keep: {sorted_files[0]}\n")
            for filepath in sorted_files[1:]:
                f.write(f"rm \"{filepath}\"\n")
            f.write("\n")
    
    print(f"\n✓ Report saved to: {output_path}")
    print(f"  - {total_duplicate_sets} duplicate sets found")
    print(f"  - {wasted_files} files could be removed")
    if cross_class_duplicates:
        print(f"  - ⚠️  {len(cross_class_duplicates)} CROSS-CLASS duplicates (check these!)")


def main():
    # Default to dataset directory
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "dataset"
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found!")
        print("Usage: python find_duplicates.py [directory]")
        sys.exit(1)
    
    print("=" * 70)
    print("DUPLICATE FILE FINDER")
    print("=" * 70)
    print(f"Scanning: {os.path.abspath(directory)}")
    print()
    
    duplicates = find_duplicates(directory)
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"duplicate_report_{timestamp}.txt"
    generate_report(duplicates, report_name, directory)


if __name__ == "__main__":
    main()
