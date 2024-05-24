import re

def is_reference_line(line):
    return r"& Pre &" in line

# Function to extract numeric values from a line
def extract_values(line):
    # 
    return list(map(float, re.findall(r'\d+\.\d+', line)))

# Function to compare lines and highlight increases
def highlight_increases(line, ref_values):
    values = extract_values(line)
    highlighted_values = []

    for value, ref_value in zip(values, ref_values):
        if value > ref_value:
            highlighted_values.append(f'\\red {value}')
        else:
            highlighted_values.append(f'{value}')
    
    # Reconstruct the line with highlighted values
    for original, highlighted in zip(values, highlighted_values):
        line = line.replace(f'{original}', highlighted, 1)

    return line

# Function to process the LaTeX file
def process_latex_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    processed_lines = []
    ref_values = None

    for line in lines:
        if is_reference_line(line):
            ref_values = extract_values(line)
            processed_lines.append(line)
        elif ref_values:
            processed_line = highlight_increases(line, ref_values)
            processed_lines.append(processed_line)
        else:
            processed_lines.append(line)

    return processed_lines

# Path to your LaTeX file
file_path = 'input.tex'

# Process the LaTeX file
processed_lines = process_latex_file(file_path)

# Write the processed lines to a new file
output_file_path = 'highlighted_latex_file.tex'
with open(output_file_path, 'w') as file:
    for line in processed_lines:
        file.write(line)

print(f'Highlighted LaTeX file saved to {output_file_path}')
