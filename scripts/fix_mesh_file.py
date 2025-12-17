input_path = "data_raw/MESH/desc2025.txt"
output_path = "data_raw/MESH/desc2025_fixed.txt"

with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# Insert a newline before every "UI = " to split records
text = text.replace(" UI = ", "\nUI = ")
text = text.replace("MN = ", "\nMN = ")
text = text.replace("MH = ", "\nMH = ")

with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"✅ Fixed file saved to {output_path}")
