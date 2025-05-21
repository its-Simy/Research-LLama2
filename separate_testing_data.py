import json

# Load the full original cleaned dialogue dataset (before flattening)
with open("educational-fine-tuning-data/cleaned_textbook1_data.json", "r") as f:
    full_data = json.load(f)

# Determine how many units to extract (last 5%)
total_units = len(full_data)
test_count = int(total_units * 0.05)  # 5%
print(f"Recovering last {test_count} units out of {total_units} total...")

# Slice last 5% as structured test set
test_units = full_data[-test_count:]

# Save to a new, safe file
output_path = "educational-fine-tuning-data/testing_structured_units.json"
with open(output_path, "w") as f:
    json.dump(test_units, f, indent=2)

print(f"✅ Saved restored test set to {output_path}")
