import audalign as ad
import os
import time
import json
import re
import csv
import pandas as pd
import numpy as np
import glob

# Create output directory for results
output_dir = "alignment_report"
os.makedirs(output_dir, exist_ok=True)

# Automatically find all offset files in dataset folder
def find_offset_files():
    offset_files = []
    
    # Search for all dataset folders
    dataset_folders = glob.glob("dataset/*")
    
    for folder in dataset_folders:
        # Look for offset_tracks subfolder
        offset_tracks_folder = os.path.join(folder, "offset_tracks")
        if os.path.exists(offset_tracks_folder) and os.path.isdir(offset_tracks_folder):
            # Find all .wav files that have "offset" in their name
            offset_files_in_folder = glob.glob(os.path.join(offset_tracks_folder, "*offset*.wav"))
            offset_files.extend(offset_files_in_folder)
    
    return offset_files

# Get list of offset files
offset_files = find_offset_files()
print(f"Found {len(offset_files)} offset files to process")

# Create temp directory for WAV files (will be deleted after each test)
temp_dir = os.path.join(output_dir, "temp")
os.makedirs(temp_dir, exist_ok=True)

# Configuration settings to test
configs = [
    {"name": "default", "bins": 84},
    {"name": "higher_res", "bins": 96},
    {"name": "lower_res", "bins": 48}
]

# Alignment methods with their class names
methods = [
    {"name": "FingerprintRecognizer", "function": ad.FingerprintRecognizer()},
    {"name": "CorrelationRecognizer", "function": ad.CorrelationRecognizer()},
    {"name": "CorrelationSpectrogramRecognizer", "function": ad.CorrelationSpectrogramRecognizer()}
]

def extract_offset_from_filename(filename):
    """Extract offset value in ms from the filename"""
    basename = os.path.basename(filename)
    
    # Try regex pattern 
    offset_pattern = re.compile(r'offset_(\d+)ms')
    match = offset_pattern.search(basename)
    if match:
        return float(match.group(1))
    
    # Try simple string split
    if "_offset_" in basename and "ms" in basename:
        try:
            offset_part = basename.split('_offset_')[1].split('ms')[0]
            return float(offset_part)
        except:
            pass
    
    return None

def extract_instrument_name(filename):
    """Extract instrument name from filename"""
    basename = os.path.basename(filename)
    
    # Handle offset files with recording device info
    if "_with_" in basename:
        basename = basename.split("_with_")[0]
    
    # Handle offset files
    if "_offset_" in basename:
        return basename.split('_offset_')[0]
    
    # Handle regular files
    return basename.split('.')[0]

def check_same_instrument(offset_file, original_file):
    """Check if the offset and original files are from the same instrument"""
    offset_base = extract_instrument_name(offset_file)
    original_base = extract_instrument_name(original_file)
    return offset_base == original_base

def get_comparison_files(offset_file):
    """Get all files to compare against for a given offset file"""
    # Get the base dataset folder
    folder = os.path.dirname(os.path.dirname(offset_file))
    
    # All files will be compared against master.wav if it exists
    master_file = os.path.join(folder, "master.wav")
    
    # Extract instrument name
    instrument = extract_instrument_name(offset_file)
    
    # Compare against original instrument file if it exists
    original_file = os.path.join(folder, f"{instrument}.wav")
    
    # Check if backing track exists (not applicable for user files)
    backing_track = None
    if not instrument.startswith("user"):
        backing_track_path = os.path.join(folder, f"no-{instrument}.wav")
        if os.path.exists(backing_track_path):
            backing_track = backing_track_path
    
    # Collect all files to compare against
    comparison_files = []
    
    # Add master file comparison if it exists
    if os.path.exists(master_file):
        comparison_files.append({
            "file": master_file,
            "type": "master"
        })
    
    # Add original file comparison if it exists
    if os.path.exists(original_file):
        comparison_files.append({
            "file": original_file,
            "type": "original"
        })
    
    # Add backing track comparison if it exists
    if backing_track and os.path.exists(backing_track):
        comparison_files.append({
            "file": backing_track,
            "type": "backing"
        })
    
    return comparison_files

def extract_device_info(filename):
    """Extract recording device information from filename"""
    basename = os.path.basename(filename)
    
    if "_with_" in basename:
        device_part = basename.split("_with_")[1].split(".")[0]
        return device_part
    
    return "Direct"  # No device info, direct recording

def run_alignment(offset_file, comparison_file, comparison_type, config, method):
    """Run alignment and return the results"""
    # Create fresh recognizer instance
    if method["name"] == "FingerprintRecognizer":
        recognizer = ad.FingerprintRecognizer()
        recognizer.config.set_accuracy(4)
    elif method["name"] == "CorrelationRecognizer":
        recognizer = ad.CorrelationRecognizer()
    elif method["name"] == "CorrelationSpectrogramRecognizer":
        recognizer = ad.CorrelationSpectrogramRecognizer()
    else:
        raise ValueError(f"Unknown recognizer: {method['name']}")
    
    # Disable multiprocessing for reliability
    if hasattr(recognizer.config, 'multiprocessing'):
        recognizer.config.multiprocessing = False
    
    # Set max lags for correlation methods
    if hasattr(recognizer.config, 'max_lags'):
        recognizer.config.max_lags = 0.5  # Allow up to 500ms of offset
    
    offset_filename = os.path.basename(offset_file)
    comparison_filename = os.path.basename(comparison_file)
    
    # Get folder
    folder = os.path.basename(os.path.dirname(os.path.dirname(offset_file)))
    
    # Get device information
    device = extract_device_info(offset_file)
    
    # Get expected offset from filename
    expected_offset_ms = extract_offset_from_filename(offset_file)
    if expected_offset_ms is None:
        print(f"Warning: Could not extract offset from {offset_filename}")
        return None
    
    # Check if same instrument
    same_instrument = check_same_instrument(offset_file, comparison_file)
    
    print(f"\nTesting {method['name']} on {offset_filename} vs {comparison_filename} ({comparison_type})")
    print(f"Config: {config['name']}, Bins: {config['bins']}")
    print(f"Expected offset: {expected_offset_ms} ms")
    print(f"Device: {device}")
    print(f"Same instrument: {same_instrument}")
    
    start_time = time.time()
    
    try:
        # Run alignment
        result = ad.align_files(
            offset_file,
            comparison_file,
            destination_path=temp_dir,
            recognizer=recognizer
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Check result
        if result and offset_filename in result:
            offset_seconds = result[offset_filename]
            if isinstance(offset_seconds, (int, float)):
                offset_ms = offset_seconds * 1000  # Allow negative values
                error_ms = offset_ms - expected_offset_ms  # Can be negative
                abs_error_ms = abs(error_ms)
                rel_error = (abs_error_ms / expected_offset_ms) * 100 if expected_offset_ms > 0 else float('inf')
                
                print(f"  Detected offset: {offset_seconds:.3f} sec ({offset_ms:.1f} ms)")
                print(f"  Error: {error_ms:.1f} ms (abs: {abs_error_ms:.1f} ms, rel: {rel_error:.2f}%)")
                print(f"  Time: {processing_time:.2f} seconds")
                
                return {
                    "folder": folder,
                    "offset_file": offset_filename,
                    "original_file": comparison_filename,
                    "comparison_type": comparison_type,
                    "device": device,
                    "true_offset": expected_offset_ms,
                    "config": config['name'],
                    "bins": config['bins'],
                    "method": method['name'],
                    "estimated_offset": offset_ms,
                    "error": error_ms,
                    "abs_error": abs_error_ms,
                    "rel_error": rel_error,
                    "same_instrument": same_instrument,
                    "processing_time": processing_time
                }
        
        print("  Failed to extract offset data from results")
        return {
            "folder": folder,
            "offset_file": offset_filename,
            "original_file": comparison_filename,
            "comparison_type": comparison_type,
            "device": device,
            "true_offset": expected_offset_ms,
            "config": config['name'],
            "bins": config['bins'],
            "method": method['name'],
            "estimated_offset": "FAILED",
            "error": "FAILED",
            "abs_error": "FAILED",
            "rel_error": "FAILED",
            "same_instrument": same_instrument,
            "processing_time": processing_time
        }
    
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"  Error: {str(e)}")
        return {
            "folder": folder,
            "offset_file": offset_filename,
            "original_file": comparison_filename,
            "comparison_type": comparison_type,
            "device": device,
            "true_offset": expected_offset_ms,
            "config": config['name'],
            "bins": config['bins'],
            "method": method['name'],
            "estimated_offset": "ERROR",
            "error": "ERROR",
            "abs_error": "ERROR",
            "rel_error": "ERROR",
            "same_instrument": same_instrument,
            "processing_time": processing_time
        }
    finally:
        # Clean temp files
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

# Print header
print("=" * 80)
print("AUDALIGN ALIGNMENT EVALUATION")
print("=" * 80)
print(f"Testing {len(offset_files)} files with multiple configurations and methods")

# Generate all tasks to run
all_tasks = []
for offset_file in offset_files:
    # Get all files to compare against
    comparison_files = get_comparison_files(offset_file)
    
    for comparison in comparison_files:
        for config in configs:
            for method in methods:
                all_tasks.append({
                    "offset_file": offset_file,
                    "comparison_file": comparison["file"],
                    "comparison_type": comparison["type"],
                    "config": config,
                    "method": method
                })

print(f"\nPrepared {len(all_tasks)} tasks to run")

# Run tasks and collect results
all_results = []
for i, task in enumerate(all_tasks):
    print(f"\nRunning task {i+1}/{len(all_tasks)}")
    result = run_alignment(
        task["offset_file"],
        task["comparison_file"],
        task["comparison_type"],
        task["config"],
        task["method"]
    )
    if result:
        all_results.append(result)

# Create DataFrame and export to CSV
df = pd.DataFrame(all_results)

# Format numerical columns
for col in ['estimated_offset', 'error', 'abs_error', 'rel_error', 'processing_time']:
    try:
        df[col] = df[col].apply(lambda x: round(float(x), 2) if isinstance(x, (int, float)) else x)
    except:
        pass

# Export to CSV
timestamp = time.strftime("%Y%m%d-%H%M%S")
csv_path = os.path.join(output_dir, f"alignment_results_{timestamp}.csv")
df.to_csv(csv_path, index=False)
print(f"\nResults exported to {csv_path}")

# Generate summary statistics
print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)

# Calculate average error by device, comparison type, method and config
summary = df.groupby(['device', 'comparison_type', 'config', 'bins', 'method']).agg({
    'abs_error': lambda x: np.mean([i for i in x if isinstance(i, (int, float))]),
    'processing_time': lambda x: np.mean([i for i in x if isinstance(i, (int, float))])
}).reset_index()

summary.columns = ['Device', 'Comparison', 'Config', 'Bins', 'Method', 'Avg Error (ms)', 'Avg Time (s)']
print("\nConfiguration Performance by Device and Comparison Type:")
print(summary.to_string(index=False))

# Find best method for each file and comparison type
print("\nBest Method For Each File and Comparison Type:")
best_methods = []

for offset_file in df['offset_file'].unique():
    for comparison_type in df['comparison_type'].unique():
        file_results = df[(df['offset_file'] == offset_file) & (df['comparison_type'] == comparison_type)]
        file_results = file_results[file_results['abs_error'].apply(lambda x: isinstance(x, (int, float)))]
        
        if not file_results.empty:
            best_row = file_results.loc[file_results['abs_error'].idxmin()]
            best_methods.append({
                'offset_file': best_row['offset_file'],
                'device': best_row['device'],
                'comparison_type': best_row['comparison_type'],
                'true_offset': best_row['true_offset'],
                'best_config': best_row['config'],
                'best_method': best_row['method'],
                'abs_error': best_row['abs_error'],
                'processing_time': best_row['processing_time']
            })

best_df = pd.DataFrame(best_methods)
print(best_df.to_string(index=False))

# Determine overall best method and config by device and comparison type
print("\nOverall Best Configuration by Device and Comparison Type:")
for device in summary['Device'].unique():
    device_summary = summary[summary['Device'] == device]
    for comparison_type in device_summary['Comparison'].unique():
        comp_summary = device_summary[device_summary['Comparison'] == comparison_type]
        if not comp_summary.empty:
            overall_best = comp_summary.loc[comp_summary['Avg Error (ms)'].idxmin()]
            print(f"\n  Device: {overall_best['Device']}")
            print(f"  Comparison: {overall_best['Comparison']}")
            print(f"  Config: {overall_best['Config']}")
            print(f"  Bins: {overall_best['Bins']}")
            print(f"  Method: {overall_best['Method']}")
            print(f"  Avg Error: {overall_best['Avg Error (ms)']:.2f} ms")
            print(f"  Avg Time: {overall_best['Avg Time (s)']:.2f} seconds")

# Export the summary to a text file
summary_path = os.path.join(output_dir, f"summary_{timestamp}.txt")
with open(summary_path, 'w') as f:
    f.write("AUDALIGN ALIGNMENT EVALUATION\n")
    f.write("=" * 80 + "\n\n")
    f.write("Configuration Performance by Device and Comparison Type:\n")
    f.write(summary.to_string(index=False) + "\n\n")
    f.write("Best Method For Each File and Comparison Type:\n")
    f.write(best_df.to_string(index=False) + "\n\n")
    f.write("Overall Best Configuration by Device and Comparison Type:\n")
    for device in summary['Device'].unique():
        device_summary = summary[summary['Device'] == device]
        for comparison_type in device_summary['Comparison'].unique():
            comp_summary = device_summary[device_summary['Comparison'] == comparison_type]
            if not comp_summary.empty:
                overall_best = comp_summary.loc[comp_summary['Avg Error (ms)'].idxmin()]
                f.write(f"\n  Device: {overall_best['Device']}\n")
                f.write(f"  Comparison: {overall_best['Comparison']}\n")
                f.write(f"  Config: {overall_best['Config']}\n")
                f.write(f"  Bins: {overall_best['Bins']}\n")
                f.write(f"  Method: {overall_best['Method']}\n")
                f.write(f"  Avg Error: {overall_best['Avg Error (ms)']:.2f} ms\n")
                f.write(f"  Avg Time: {overall_best['Avg Time (s)']:.2f} seconds\n")

print(f"\nSummary exported to {summary_path}") 