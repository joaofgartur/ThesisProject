import re


def pad_floats_to_4_decimals(latex_table):
    # Define a������ pattern to match floats
    float_pattern = r'\d+\.\d+'
    # Find all floats in the LaTeX table
    floats = re.findall(float_pattern, latex_table)
    # Pad each float to 4 decimals
    padded_floats = [f'{float(float_str):.4f}' for float_str in floats]
    # Replace the floats in the LaTeX table with the padded ones
    padded_latex_table = re.sub(float_pattern, lambda x: padded_floats.pop(0), latex_table)
    return padded_latex_table


def process_latex_table(input_file, output_file):
    # Read LaTeX table from input file
    with open(input_file, 'r') as f:
        latex_table = f.read()

    # Pad floats to 4 decimals
    padded_latex_table = pad_floats_to_4_decimals(latex_table)

    # Write modified LaTeX table to output file
    with open(output_file, 'w') as f:
        f.write(padded_latex_table)


# Input and output file paths
input_file = 'input.tex'
output_file = 'output.tex'

# Process LaTeX table and write to output file
process_latex_table(input_file, output_file)
