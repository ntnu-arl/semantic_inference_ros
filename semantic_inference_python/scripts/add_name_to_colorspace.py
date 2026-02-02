import pandas as pd
from matplotlib import colors
import webcolors
from pathlib import Path
import argparse

# Argument parser for input CSV file
parser = argparse.ArgumentParser(description="Add color names to colorspace CSV.")
parser.add_argument("--input_csv", type=str, required=True, help="Path to the input colorspace CSV file.")
parser.add_argument("--output_csv", type=str, required=True, help="Path to the output CSV file with color names.")
args = parser.parse_args()

input_csv_path = Path(args.input_csv)
output_csv_path = Path(args.output_csv)
assert input_csv_path.exists(), f"Input CSV file {input_csv_path} does not exist."

# Read CSV into dataframe
df = pd.read_csv(input_csv_path)

# Function to find nearest color name
def closest_color_name(requested_color):
    min_colours = {}
    webcolors.hex_to_name
    for key, name in webcolors._definitions._get_hex_to_name_map(webcolors.CSS3).items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_color_name(r, g, b):
    try:
        name = webcolors.rgb_to_name((r, g, b))
    except:
        name = closest_color_name((r, g, b))
    return name.replace("-", " ")

# Generate color names ensuring uniqueness
used_names = set()
color_names = []

for _, row in df.iterrows():
    base_name = get_color_name(row.red, row.green, row.blue)
    name = base_name
    suffix = 1
    while name in used_names:
        name = f"{base_name} {suffix}"
        suffix += 1
    used_names.add(name)
    color_names.append(name)

df["color"] = color_names

# Save to CSV file
df.to_csv(output_csv_path, index=False)

