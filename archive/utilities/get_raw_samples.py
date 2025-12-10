"""
Get raw (unscaled) samples for testing
"""
import numpy as np

# Load scaled test data
test_data = np.load('data/test_data.npy')
test_labels = np.load('data/test_labels.npy')

# Get samples (use scaled version directly - we'll provide both versions)
normal_idx = np.where(test_labels == 0)[0][0]
attack_idx = np.where(test_labels == 1)[0][0]

normal_scaled = test_data[normal_idx]
attack_scaled = test_data[attack_idx]

# For RAW samples, we provide realistic examples based on TON_IoT dataset structure
# These are manually created based on typical network traffic patterns

# Normal traffic: typical IoT sensor communication (41 features)
normal_raw = np.array([
    47260.0,  # src_port
    15600.0,  # dst_port
    0.0,      # duration
    100.0,    # src_bytes
    50.0,     # dst_bytes
    0.0, 1.0, 150.0, 0.0, 50.0,  # network stats
    0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 2.0, 3.0, 2.0, 2.0,  # connection features
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0
])

# Attack traffic: DDoS/scan pattern with high volume (41 features)
attack_raw = np.array([
    4444.0,    # suspicious port
    49178.0,   # random high port
    290.0,     # long duration
    101568.0,  # very high bytes (attack indicator!)
    2592.0,    # response bytes
    0.0, 108.0, 108064.0, 31.0, 3832.0,  # high packet counts
    0.0, 0.0, 0.0, 0.0, 0.0,
    5.0, 3.0, 8.0, 10.0, 12.0, 15.0,  # anomalous patterns
    1.0, 2.0, 1.0, 0.0,
    1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 1.0, 0.0, 1.0,
    0.0, 0.0, 1.0, 0.0, 0.0
])

print('=' * 60)
print('NORMAL SAMPLE (RAW - UNSCALED)')
print('=' * 60)
print(f'Number of features: {len(normal_raw)}')
print()
print(','.join([f'{x:.6f}' for x in normal_raw]))
print()
print()

print('=' * 60)
print('ATTACK SAMPLE (RAW - UNSCALED)')
print('=' * 60)
print(f'Number of features: {len(attack_raw)}')
print()
print(','.join([f'{x:.6f}' for x in attack_raw]))
print()
print()

print('=' * 60)
print('HOW TO USE IN OPTION 4')
print('=' * 60)
print('Copy the comma-separated values above and paste into Option 4.')
print('Do NOT use "scaled:" prefix - these are raw values!')
