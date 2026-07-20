"""CLI for JSON prettifier/minifier with multiprocessing support."""

import argparse
import json
import mmap
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

# Threshold for using mmap (5MB)
MMAP_THRESHOLD = 5 * 1024 * 1024  # 5MB in bytes


def process_json_file(args_tuple):
    """Process a single JSON file - prettify or minify.
    
    Args:
        args_tuple: Tuple containing (file_path, minify, sort_keys)
    
    Returns:
        Tuple of (file_path, success, error_message)
    """
    file_path, minify, sort_keys = args_tuple
    file_path = Path(file_path)
    
    try:
        # Get file size
        file_size = file_path.stat().st_size
        
        # Choose reading method based on file size
        if file_size > MMAP_THRESHOLD:
            # Use mmap for large files
            with open(file_path, 'r+b') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    # Read the entire content as bytes and decode
                    content = mmapped_file.read().decode('utf-8')
                    data = json.loads(content)
        else:
            # Use regular file reading for smaller files
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        # Determine output parameters
        indent = None if minify else 2
        
        # Write the processed JSON back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(
                data,
                f,
                indent=indent,
                sort_keys=sort_keys,
                ensure_ascii=False
            )
        
        return (str(file_path), True, None)
    
    except json.JSONDecodeError as e:
        return (str(file_path), False, f"Invalid JSON: {e}")
    except Exception as e:
        return (str(file_path), False, f"Error: {e}")


def collect_json_files(paths, recursive=True):
    """Collect all JSON files from given paths.
    
    Args:
        paths: List of file/directory paths
        recursive: Whether to search directories recursively
    
    Returns:
        Set of Path objects for JSON files
    """
    json_files = set()
    
    for path in paths:
        path = Path(path)
        
        if path.is_file():
            if path.suffix.lower() == '.json':
                json_files.add(path)
        elif path.is_dir():
            if recursive:
                pattern = '**/*.json'
                json_files.update(path.glob(pattern))
            else:
                json_files.update(path.glob('*.json'))
    
    return json_files


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Prettify or minify JSON files with multiprocessing support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  jb                    # Prettify all JSON files in current directory recursively
  jb file.json         # Prettify specific file
  jb -m file.json      # Minify specific file
  jb -s dir/           # Prettify with sorted keys
  jb -m -s file.json   # Minify with sorted keys
  jb file1.json dir1/  # Process multiple files and directories
        """
    )
    
    # Create mutually exclusive group for beautify/minify
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '-b', '--beautify',
        action='store_true',
        default=True,
        help='Beautify/prettify JSON (default)'
    )
    mode_group.add_argument(
        '-m', '--minify',
        action='store_true',
        help='Minify JSON (remove whitespace)'
    )
    
    parser.add_argument(
        '-s', '--sort-keys',
        action='store_true',
        default=False,
        help='Sort JSON keys alphabetically (default: False)'
    )
    
    parser.add_argument(
        'paths',
        nargs='*',
        default=None,
        help='JSON files or directories to process (default: current directory)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help=f'Number of worker processes (default: CPU count = {cpu_count()})'
    )
    
    args = parser.parse_args()
    
    # If no paths provided, use current directory
    if not args.paths:
        args.paths = ['.']
    
    # Determine mode
    minify = args.minify  # If -m is set, minify; otherwise beautify
    
    # Collect JSON files
    json_files = collect_json_files(args.paths, recursive=True)
    
    if not json_files:
        print("No JSON files found.", file=sys.stderr)
        sys.exit(1)
    
    # Prepare arguments for multiprocessing
    process_args = [(f, minify, args.sort_keys) for f in sorted(json_files)]
    
    # Determine number of workers
    workers = args.workers or cpu_count()
    workers = min(workers, len(process_args))
    
    print(f"Processing {len(json_files)} JSON file(s) using {workers} worker(s)...")
    mode_text = "Minifying" if minify else "Prettifying"
    sort_text = " with sorted keys" if args.sort_keys else ""
    print(f"{mode_text}{sort_text}...")
    
    # Process files with multiprocessing
    with Pool(processes=workers) as pool:
        results = pool.map(process_json_file, process_args)
    
    # Report results
    success_count = 0
    error_count = 0
    
    for file_path, success, error in results:
        if success:
            success_count += 1
        else:
            error_count += 1
            print(f"✗ {file_path}: {error}", file=sys.stderr)
    
    print(f"\n✓ Successfully processed: {success_count} file(s)")
    if error_count > 0:
        print(f"✗ Failed: {error_count} file(s)")
        sys.exit(1)
    else:
        print("All files processed successfully!")


if __name__ == '__main__':
    main()