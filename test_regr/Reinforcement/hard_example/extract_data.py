import json
import sys

if len(sys.argv) < 2:
    print('Usage: python extract_json_data.py <input_file> [key_to_extract]')
    print('Example: python extract_json_data.py data.json')
    print('Example: python extract_json_data.py data.json entities_only_with_1_things_YN')
    sys.exit(1)

INPUT_FILE = sys.argv[1]

# Read the large JSON file
print(f'Reading input file: {INPUT_FILE}')
with open(INPUT_FILE, 'r') as f:
    data = json.load(f)

if len(sys.argv) < 3:
    print('\nAvailable keys in the JSON file:')
    print('-' * 50)
    for i, key in enumerate(data.keys(), 1):
        print(f'{i}. {key}')
    print('-' * 50)
    print(f'\nUsage: python extract_json_data.py {INPUT_FILE} <key_to_extract>')
    print(f'Example: python extract_json_data.py {INPUT_FILE} entities_only_with_1_things_YN')
    sys.exit(0)

KEY_TO_EXTRACT = sys.argv[2]
OUTPUT_FILE = f'{KEY_TO_EXTRACT}.json'

# Extract specified key
print(f'Extracting key: {KEY_TO_EXTRACT}')
if KEY_TO_EXTRACT in data:
    extracted = {KEY_TO_EXTRACT: data[KEY_TO_EXTRACT]}
    print(f'  ✓ Extracted: {KEY_TO_EXTRACT}')
else:
    print(f'  ✗ Key not found: {KEY_TO_EXTRACT}')
    sys.exit(1)

# Write to output file
print(f'Writing output file...')
with open(OUTPUT_FILE, 'w') as f:
    json.dump(extracted, f, indent=2)

print(f'Done! Extracted to {OUTPUT_FILE}')