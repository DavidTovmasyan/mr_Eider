import pickle
import os

# Define the base file path
base_path = "/home/david/david_tovmasyan/mr_Eider/saved_features/eider_rule_hoi_train_distant_{}.pkl"

# List to store data
combined_data = []

# Load each file sequentially
for i in range(10):  # Assuming IDs are from 0 to 9
    file_path = base_path.format(i)
    
    # Check if the file exists (optional, but recommended)
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            combined_data.extend(data)  # Append preserving order
    else:
        print(f"Warning: {file_path} not found!")

# Save the combined data
output_path = "/home/david/david_tovmasyan/mr_Eider/saved_features/eider_rule_hoi_train_distant.pkl"
with open(output_path, "wb") as f:
    pickle.dump(combined_data, f)

print(f"Combined file saved at: {output_path}")
