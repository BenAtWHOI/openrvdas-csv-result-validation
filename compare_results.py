import re
import numpy as np
from datetime import datetime
from scipy import interpolate
import matplotlib.pyplot as plt


def parse_gyro_line(line):
    """Parse a single line of gyro data"""
    # Extract timestamp and HEHDT value
    # Format: NAV 2025/06/18 14:00:00.828 GYRO $HEHDT,177.92,*T*15*
    parts = line.strip().split()
    if len(parts) < 4 or 'HEHDT' not in line:
        return None, None

    timestamp_str = f"{parts[1]} {parts[2]}"
    timestamp = datetime.strptime(timestamp_str, "%Y/%m/%d %H:%M:%S.%f")

    # Extract heading value from $HEHDT,177.92,*T*15*
    hehdt_match = re.search(r'HEHDT,(\d+\.\d+)', line)
    if hehdt_match:
        heading = float(hehdt_match.group(1))
        return timestamp, heading

    return None, None


def load_gyro_file(filename):
    """Load gyro data from file"""
    timestamps = []
    headings = []

    with open(filename, 'r') as f:
        for line in f:
            timestamp, heading = parse_gyro_line(line)
            if timestamp is not None and heading is not None:
                timestamps.append(timestamp)
                headings.append(heading)

    return np.array(timestamps), np.array(headings)


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


def plot_comparison(ref_times, ref_values, target_values, metrics):
    """Create comparison plots"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Time series plot
    ax1.plot(ref_times, ref_values, 'b-', label='Reference Stream', alpha=0.8)
    ax1.plot(ref_times, target_values, 'r--', label='Target Stream (interpolated)', alpha=0.8)
    ax1.set_xlabel('Time (seconds from start)')
    ax1.set_ylabel('Heading (degrees)')
    ax1.set_title('Time Series Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Difference plot
    diff = ref_values - target_values
    ax2.plot(ref_times, diff, 'g-', alpha=0.8)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (seconds from start)')
    ax2.set_ylabel('Difference (degrees)')
    ax2.set_title(
        f'Difference (Reference - Target)\nMAE: {metrics["mae"]:.3f}°, RMSE: {metrics["rmse"]:.3f}°, Bias: {metrics["bias"]:.3f}°')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    dslog_stream = './data/dslog_at20250618_1400.HDT'
    openrvdas_stream = './data/openrvdas_at20250618_1400.HDT'

    print("Loading gyro data files...")

    try:
        ref_times, ref_values = load_gyro_file(dslog_stream)
        target_times, target_values = load_gyro_file(openrvdas_stream)

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

        print("\n" + "="*50)
        print("COMPARISON RESULTS")
        print("="*50)
        print(f"Number of compared points: {metrics['n_points']}")
        print(f"Mean Absolute Error (MAE): {metrics['mae']:.3f}°")
        print(f"Root Mean Square Error (RMSE): {metrics['rmse']:.3f}°")
        print(f"Bias (Ref - Target): {metrics['bias']:.3f}°")
        print(f"Correlation: {metrics['correlation']:.3f}")

        plot_comparison(time_aligned, ref_aligned, target_aligned, metrics)

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
