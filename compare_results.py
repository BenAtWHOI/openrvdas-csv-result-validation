import re
import argparse
import numpy as np
from datetime import datetime
from scipy import interpolate
import matplotlib.pyplot as plt

# Predefined configurations for known data formats
CONFIGS = {
    'GYRO': {
        'description': 'GYRO heading data ($HEHDT)',
        'pattern': r'NAV\s+(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2}\.\d+)\s+GYRO\s+\$HEHDT,(\d+\.\d+)',
        'fields': ['heading'],
        'units': {'heading': 'degrees'},
        'timestamp_groups': [1, 2],  # Date and time group indices
        'data_groups': [3],  # Heading value group index
        'timestamp_format': '%Y/%m/%d %H:%M:%S.%f'
    },
    'SBE45': {
        'description': 'SBE45 CTD data (temp, salinity, oxygen, sound velocity)',
        'pattern': r'SSW\s+(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2}\.\d+)\s+SBE45\s+(\S+),\s+(\S+),\s+(\S+),\s+(\S+)',
        'fields': ['temperature', 'salinity', 'oxygen', 'sound_velocity'],
        'units': {
            'temperature': '°C',
            'salinity': 'PSU',
            'oxygen': 'mg/L',
            'sound_velocity': 'm/s'
        },
        'timestamp_groups': [1, 2],
        'data_groups': [3, 4, 5, 6],
        'timestamp_format': '%Y/%m/%d %H:%M:%S.%f'
    },
    'GPS': {
        'description': 'GPS position data ($GPGGA)',
        'pattern': r'NAV\s+(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2}\.\d+)\s+GPS\s+\$GPGGA,[^,]*,([^,]*),([NS]),([^,]*),([EW])',
        'fields': ['latitude', 'longitude'],
        'units': {
            'latitude': 'degrees',
            'longitude': 'degrees'
        },
        'timestamp_groups': [1, 2],
        'data_groups': [3, 4, 5, 6],  # lat, lat_dir, lon, lon_dir
        'timestamp_format': '%Y/%m/%d %H:%M:%S.%f',
        'custom_parser': 'gps'
    }
}


def parse_gps_position(groups):
    """Custom parser for GPS position data"""
    lat_deg = float(groups[0][:2]) if groups[0] else 0
    lat_min = float(groups[0][2:]) if groups[0] else 0
    lat = lat_deg + lat_min / 60
    if groups[1] == 'S':
        lat = -lat

    lon_deg = float(groups[2][:3]) if groups[2] else 0
    lon_min = float(groups[2][3:]) if groups[2] else 0
    lon = lon_deg + lon_min / 60
    if groups[3] == 'W':
        lon = -lon

    return [lat, lon]


def parse_line(line, config):
    """Parse a single line of data based on configuration"""
    pattern = re.compile(config['pattern'])
    match = pattern.search(line)

    if not match:
        return None, None

    groups = match.groups()

    # Extract timestamp
    timestamp_parts = [groups[i] for i in config['timestamp_groups']]
    timestamp_str = ' '.join(timestamp_parts)
    try:
        timestamp = datetime.strptime(timestamp_str, config['timestamp_format'])
    except ValueError as e:
        print(f"Debug: Error parsing timestamp")
        print(f"  Line: {line[:80]}...")
        print(f"  Pattern: {config['pattern']}")
        print(f"  Groups: {groups}")
        print(f"  Timestamp parts: {timestamp_parts}")
        print(f"  Timestamp string: '{timestamp_str}'")
        print(f"  Format: {config['timestamp_format']}")
        raise ValueError(f"time data '{timestamp_str}' does not match format '{config['timestamp_format']}'")

    # Extract data values
    data_groups = [groups[i] for i in config['data_groups']]

    # Apply custom parser if available
    if 'custom_parser' in config:
        if config['custom_parser'] == 'gps':
            values = parse_gps_position(data_groups)
        # Add more custom parsers here as needed
    else:
        values = [float(g) for g in data_groups]

    # Create data dictionary
    data = dict(zip(config['fields'], values))

    return timestamp, data


def load_stream_file(filename, config):
    """Load stream data from file"""
    timestamps = []
    data_records = []

    with open(filename, 'r') as f:
        for line in f:
            timestamp, data = parse_line(line, config)
            if timestamp is not None and data is not None:
                timestamps.append(timestamp)
                data_records.append(data)

    return np.array(timestamps), data_records


def interpolate_to_reference(ref_times, ref_data, target_times, target_data, compare_field):
    """Interpolate target data to reference timestamps"""
    # Extract the comparison field values
    ref_values = np.array([d[compare_field] for d in ref_data])
    target_values = np.array([d[compare_field] for d in target_data])

    # Convert timestamps to seconds from first timestamp
    ref_start = ref_times[0]
    ref_seconds = np.array([(t - ref_start).total_seconds() for t in ref_times])
    target_seconds = np.array([(t - ref_start).total_seconds() for t in target_times])

    # Find overlapping time range
    min_time = max(ref_seconds.min(), target_seconds.min())
    max_time = min(ref_seconds.max(), target_seconds.max())

    # Filter reference data to overlapping range
    valid_ref_mask = (ref_seconds >= min_time) & (ref_seconds <= max_time)
    valid_ref_times = ref_seconds[valid_ref_mask]
    valid_ref_values = ref_values[valid_ref_mask]

    # Create interpolation function for target data
    interp_func = interpolate.interp1d(target_seconds, target_values,
                                       kind='linear', bounds_error=False, fill_value=np.nan)

    # Interpolate target values at reference timestamps
    interpolated_values = interp_func(valid_ref_times)

    return valid_ref_values, interpolated_values, valid_ref_times


def calculate_metrics(ref_values, target_values):
    """Calculate comparison metrics"""
    # Remove any NaN values
    valid_mask = ~(np.isnan(ref_values) | np.isnan(target_values))
    ref_clean = ref_values[valid_mask]
    target_clean = target_values[valid_mask]

    if len(ref_clean) == 0:
        return None

    # Calculate metrics
    diff = ref_clean - target_clean
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    bias = np.mean(diff)
    correlation = np.corrcoef(ref_clean, target_clean)[0, 1]

    return {
        'mae': mae,
        'rmse': rmse,
        'bias': bias,
        'correlation': correlation,
        'n_points': len(ref_clean),
        'max_diff': np.max(np.abs(diff)),
        'std_diff': np.std(diff)
    }


def plot_comparison(ref_times, ref_values, target_values, metrics, config, compare_field, data_type):
    """Create comparison plots"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    unit = config['units'].get(compare_field, '')
    field_label = f"{compare_field.replace('_', ' ').title()} ({unit})" if unit else compare_field.replace(
        '_', ' ').title()

    # Time series plot
    ax1.plot(ref_times, ref_values, 'b-', label='Reference Stream', alpha=0.8, linewidth=1)
    ax1.plot(ref_times, target_values, 'r--',
             label='Target Stream (interpolated)', alpha=0.8, linewidth=1)
    ax1.set_xlabel('Time (seconds from start)')
    ax1.set_ylabel(field_label)
    ax1.set_title(f'{data_type} Stream Comparison - {field_label}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Difference plot
    diff = ref_values - target_values
    ax2.plot(ref_times, diff, 'g-', alpha=0.8)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.fill_between(ref_times, 0, diff, alpha=0.3, color='g')
    ax2.set_xlabel('Time (seconds from start)')
    ax2.set_ylabel(f'Difference ({unit})' if unit else 'Difference')
    ax2.set_title(f'Difference (Reference - Target)\n'
                  f'MAE: {metrics["mae"]:.4f}, RMSE: {metrics["rmse"]:.4f}, '
                  f'Bias: {metrics["bias"]:.4f}, Max: {metrics["max_diff"]:.4f}')
    ax2.grid(True, alpha=0.3)

    # Scatter plot
    ax3.scatter(ref_values, target_values, alpha=0.5, s=1)
    # Add perfect agreement line
    min_val = min(ref_values.min(), target_values.min())
    max_val = max(ref_values.max(), target_values.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Agreement')
    ax3.set_xlabel(f'Reference {field_label}')
    ax3.set_ylabel(f'Target {field_label}')
    ax3.set_title(f'Scatter Plot (Correlation: {metrics["correlation"]:.4f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def compare_streams(data_type, ref_file, target_file, compare_field=None):
    """Main comparison function"""
    # Validate data type
    if data_type not in CONFIGS:
        raise ValueError(f"Unknown data type: {data_type}. Available types: {list(CONFIGS.keys())}")

    config = CONFIGS[data_type]

    # Set comparison field
    if compare_field:
        if compare_field not in config['fields']:
            raise ValueError(f"Field '{compare_field}' not available for {data_type}. "
                             f"Available fields: {config['fields']}")
    else:
        # Default to first field
        compare_field = config['fields'][0]

    print(f"\nComparing {data_type} streams")
    print(f"Field: {compare_field}")
    print(f"Reference: {ref_file}")
    print(f"Target: {target_file}")
    print("-" * 60)

    # Load data
    print("Loading stream data files...")
    ref_times, ref_data = load_stream_file(ref_file, config)
    target_times, target_data = load_stream_file(target_file, config)

    print(f"Reference stream: {len(ref_data)} points")
    print(f"Target stream: {len(target_data)} points")

    if len(ref_data) == 0 or len(target_data) == 0:
        print("Error: One or both files contain no valid data")
        return None

    # Interpolate and align
    print("Interpolating target stream to reference timestamps...")
    ref_aligned, target_aligned, time_aligned = interpolate_to_reference(
        ref_times, ref_data, target_times, target_data, compare_field)

    print(f"Aligned data: {len(ref_aligned)} points")

    # Calculate metrics
    metrics = calculate_metrics(ref_aligned, target_aligned)

    if metrics is None:
        print("Error: No valid overlapping data points found")
        return None

    # Display results
    unit = config['units'].get(compare_field, '')
    unit_str = f" {unit}" if unit else ""

    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Data Type: {data_type} - {config['description']}")
    print(f"Compared Field: {compare_field}{unit_str}")
    print(f"Number of compared points: {metrics['n_points']:,}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.6f}{unit_str}")
    print(f"Root Mean Square Error (RMSE): {metrics['rmse']:.6f}{unit_str}")
    print(f"Bias (Ref - Target): {metrics['bias']:.6f}{unit_str}")
    print(f"Standard Deviation of Diff: {metrics['std_diff']:.6f}{unit_str}")
    print(f"Maximum Difference: {metrics['max_diff']:.6f}{unit_str}")
    print(f"Correlation: {metrics['correlation']:.6f}")
    print("="*60)

    # Create plots
    plot_comparison(time_aligned, ref_aligned, target_aligned, metrics,
                    config, compare_field, data_type)

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_type', nargs='?',
                        help='Type of data to compare (GYRO, SBE45, GPS)')
    parser.add_argument('reference_file', nargs='?',
                        help='Reference stream file')
    parser.add_argument('target_file', nargs='?',
                        help='Target stream file to compare against reference')
    parser.add_argument('--field', '-f', dest='compare_field',
                        help='Specific field to compare (default: first available field)')

    args = parser.parse_args()

    if not args.data_type or not args.reference_file or not args.target_file:
        parser.error("data_type, reference_file, and target_file are required")

    try:
        compare_streams(args.data_type.upper(), args.reference_file,
                        args.target_file, args.compare_field)
    except ValueError as e:
        print(f"ValueError: {e}")
        return
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

    return


if __name__ == "__main__":
    exit(main())
