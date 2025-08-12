import re
import sys
import argparse
import numpy as np
from datetime import datetime
from scipy import interpolate
import matplotlib.pyplot as plt


# Parser configurations for different data formats
PARSER_CONFIGS = {
    'gyro': {
        'pattern': r'(\S+)\s+(\d{4}/\d{2}/\d{2})\s+([\d:.]+)\s+GYRO.*HEHDT,(\d+\.\d+)',
        'timestamp_format': '%Y/%m/%d %H:%M:%S.%f',
        'timestamp_groups': (2, 3),  # date and time groups in regex
        'fields': ['heading'],
        'field_extractors': {
            'heading': lambda match: float(match.group(4))
        },
        'display_name': 'Heading',
        'units': 'degrees'
    },
    'sbe45': {
        'pattern': r'(\S+)\s+(\d{4}/\d{2}/\d{2})\s+([\d:.]+)\s+SBE45\s+([\d.]+),\s+([\d.]+),\s+([\d.]+),\s+([\d.]+)',
        'timestamp_format': '%Y/%m/%d %H:%M:%S.%f',
        'timestamp_groups': (2, 3),
        'fields': ['temperature', 'salinity', 'oxygen', 'sound_velocity'],
        'field_extractors': {
            'temperature': lambda match: float(match.group(4)),
            'salinity': lambda match: float(match.group(5)),
            'oxygen': lambda match: float(match.group(6)),
            'sound_velocity': lambda match: float(match.group(7))
        },
        'display_name': 'Temperature',  # Default to first field
        'units': '°C'
    }
}


class GenericDataParser:
    """Generic parser for different data formats based on configuration"""

    def __init__(self, data_type, config=None):
        """Initialize parser with configuration for specific data type"""
        self.data_type = data_type
        self.config = config or PARSER_CONFIGS.get(data_type)
        if not self.config:
            raise ValueError(f"No configuration found for data type: {data_type}")

        self.pattern = re.compile(self.config['pattern'])
        self.fields = self.config['fields']
        self.field_extractors = self.config['field_extractors']

    def parse_line(self, line):
        """Parse a single line of data according to configuration"""
        match = self.pattern.search(line.strip())
        if not match:
            return None, None

        # Extract timestamp
        date_group, time_group = self.config['timestamp_groups']
        timestamp_str = f"{match.group(date_group)} {match.group(time_group)}"
        try:
            timestamp = datetime.strptime(timestamp_str, self.config['timestamp_format'])
        except ValueError:
            return None, None

        # Extract field values
        values = {}
        for field in self.fields:
            try:
                extractor = self.field_extractors[field]
                values[field] = extractor(match)
            except (ValueError, IndexError):
                return None, None

        return timestamp, values


def parse_gyro_line(line):
    """Legacy function for backward compatibility - parse a single line of gyro data"""
    parser = GenericDataParser('gyro')
    timestamp, values = parser.parse_line(line)
    if timestamp and values:
        return timestamp, values['heading']
    return None, None


def load_data_file(filename, data_type='gyro', field_index=0):
    """Load data from file using appropriate parser

    Args:
        filename: Path to data file
        data_type: Type of data ('gyro', 'sbe45', etc.)
        field_index: Index of field to extract (default: 0 for first field)

    Returns:
        timestamps: Array of datetime objects
        values: Array of values for the specified field
        field_name: Name of the field being returned
    """
    parser = GenericDataParser(data_type)
    timestamps = []
    all_values = {field: [] for field in parser.fields}

    with open(filename, 'r') as f:
        for line in f:
            timestamp, values = parser.parse_line(line)
            if timestamp is not None and values is not None:
                timestamps.append(timestamp)
                for field, value in values.items():
                    all_values[field].append(value)

    # Return specified field (default to first)
    field_name = parser.fields[field_index]
    return np.array(timestamps), np.array(all_values[field_name]), field_name


def load_gyro_file(filename):
    """Legacy function for backward compatibility"""
    timestamps, values, _ = load_data_file(filename, 'gyro', 0)
    return timestamps, values


def interpolate_to_reference(ref_times, ref_values, target_times, target_values):
    """Interpolate target data to reference timestamps"""
    # Convert timestamps to seconds from first timestamp for easier interpolation
    ref_start = ref_times[0]
    ref_seconds = np.array([(t - ref_start).total_seconds() for t in ref_times])
    target_seconds = np.array([(t - ref_start).total_seconds() for t in target_times])

    # Create interpolation function
    # Only interpolate within the overlapping time range
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
        'n_points': len(ref_clean)
    }


def plot_comparison(ref_times, ref_values, target_values, metrics, data_type='gyro', field_name=None):
    """Create comparison plots"""
    # Get display configuration
    config = PARSER_CONFIGS.get(data_type, {})
    display_name = field_name or config.get('display_name', 'Value')
    units = config.get('units', '')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Time series plot
    ax1.plot(ref_times, ref_values, 'b-', label='Reference Stream', alpha=0.8)
    ax1.plot(ref_times, target_values, 'r--', label='Target Stream (interpolated)', alpha=0.8)
    ax1.set_xlabel('Time (seconds from start)')
    ax1.set_ylabel(f'{display_name} ({units})' if units else display_name)
    ax1.set_title(f'Time Series Comparison - {display_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Difference plot
    diff = ref_values - target_values
    ax2.plot(ref_times, diff, 'g-', alpha=0.8)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (seconds from start)')
    ax2.set_ylabel(f'Difference ({units})' if units else 'Difference')

    # Format metrics display with appropriate units
    units_str = f' {units}' if units else ''
    ax2.set_title(
        f'Difference (Reference - Target)\nMAE: {metrics["mae"]:.3f}{units_str}, RMSE: {metrics["rmse"]:.3f}{units_str}, Bias: {metrics["bias"]:.3f}{units_str}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare two data streams')
    parser.add_argument('reference', help='Path to reference data file')
    parser.add_argument('target', help='Path to target data file')
    parser.add_argument('--type', default='gyro', choices=list(PARSER_CONFIGS.keys()),
                        help='Type of data to parse (default: gyro)')
    parser.add_argument('--field', type=int, default=0,
                        help='Field index to compare (default: 0 for first field)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting the comparison')

    args = parser.parse_args()

    data_type = args.type
    field_index = args.field

    print(f"Loading {data_type} data files...")

    try:
        # Load data files
        ref_times, ref_values, field_name = load_data_file(args.reference, data_type, field_index)
        target_times, target_values, _ = load_data_file(args.target, data_type, field_index)

        print(f"Data type: {data_type}")
        print(f"Comparing field: {field_name}")
        print(f"Reference stream: {len(ref_values)} points")
        print(f"Target stream: {len(target_values)} points")

        print("Interpolating target stream to reference timestamps...")
        ref_aligned, target_aligned, time_aligned = interpolate_to_reference(
            ref_times, ref_values, target_times, target_values)

        print(f"Aligned data: {len(ref_aligned)} points")

        metrics = calculate_metrics(ref_aligned, target_aligned)

        if metrics is None:
            print("Error: No valid overlapping data points found")
            return

        # Get units for display
        config = PARSER_CONFIGS[data_type]
        units = config.get('units', '')
        units_str = f' {units}' if units else ''

        print("\n" + "="*50)
        print("COMPARISON RESULTS")
        print("="*50)
        print(f"Data type: {data_type}")
        print(f"Field: {field_name}")
        print(f"Number of compared points: {metrics['n_points']}")
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.3f}{units_str}")
        print(f"Root Mean Square Error (RMSE): {metrics['rmse']:.3f}{units_str}")
        print(f"Bias (Ref - Target): {metrics['bias']:.3f}{units_str}")
        print(f"Correlation: {metrics['correlation']:.3f}")

        if not args.no_plot:
            plot_comparison(time_aligned, ref_aligned, target_aligned,
                            metrics, data_type, field_name)

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
