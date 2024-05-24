import re
from argparse import ArgumentParser


float_pattern = r'\d+\.\d+'


def read_file(input_file, all=True):
    with open(input_file, 'r') as f:
        if all:
            content = f.read()
        else:
            content = f.readlines()
    return content


def write_file(content, output_file, all=True):
    with open(output_file, 'w') as f:
        if all:
            f.write(content)
        else:
            for line in content:
                f.write(line)


def pad_floats(latex_table, num_decimals):
    floats = re.findall(float_pattern, latex_table)
    padded_floats = [f'{float(float_str):.{num_decimals}f}' for float_str in floats]
    padded_latex_table = re.sub(float_pattern, lambda x: padded_floats.pop(0), latex_table)
    return padded_latex_table


def is_reference_line(line):
    return r"& Pre &" in line


def extract_values(line):
    values = list(map(float, re.findall(float_pattern, line)))
    return values


def highlight_ref_line(line):
    floats = re.findall(float_pattern, line)
    highlighted_floats = [f'\\gray {float(float_str)}' for float_str in floats]
    padded_line = re.sub(float_pattern, lambda x: highlighted_floats.pop(0), line)
    return padded_line


def highlight_increases(line, ref_values):
    values = extract_values(line)
    highlighted_values = []

    for value, ref_value in zip(values, ref_values):
        if value > ref_value:
            highlighted_values.append(f'\\red {value}')
        else:
            highlighted_values.append(f'{value}')
    
    for original, highlighted, ref_value in zip(values, highlighted_values, ref_values):
        if original > ref_value:
            line = line.replace(f'{original}', highlighted, 1)

    return line


def highlight_accepted(line, target):
    floats = re.findall(float_pattern, line)
    highlighted_floats = [f'\\green {float(float_str)}' if float(float_str) <= target else f'{float(float_str)}' for float_str in floats]
    padded_line = re.sub(float_pattern, lambda x: highlighted_floats.pop(0), line)
    return padded_line

def highlight_non_ref_line(line, ref_values, target):
    line = highlight_accepted(line, target)
    return highlight_increases(line, ref_values)


def highlight_table(lines, target):

    processed_table = []
    ref_values = None

    for line in lines:
        if is_reference_line(line):
            ref_values = extract_values(line)
            line = highlight_ref_line(line)
        elif ref_values:
            line = highlight_non_ref_line(line, ref_values, target)
        processed_table.append(line)

    processed_table = fix_highlight_errors(processed_table)

    return processed_table


def fix_highlight_errors(lines):
    for i, line in enumerate(lines):
        if all(float(value) == 0 for value in re.findall(r'\d+\.\d+', line)):
            line = re.sub(r'\\green', r'\\red', line)
        lines[i] = re.sub(r'\\red \\red\s+', r'\\red ', line)
        lines[i] = re.sub(r'\\green \\green\s+', r'\\red ', line)
        lines[i] = re.sub(r'\\green \\red', r'\\green', line)
    return lines


def is_data_line(line):
    has_floats = extract_values(line)
    return has_floats and line.strip().startswith("&")


def compare_pairs(pair1, pair2):
    increased, decreased, unchanged = 0, 0, 0
    values1 = extract_values(pair1)
    values2 = extract_values(pair2)
    
    for v1, v2 in zip(values1, values2):
        if v2 > v1:
            increased += 1
        elif v2 < v1:
            decreased += 1
        else:
            unchanged += 1
    
    return increased, decreased, unchanged


def process_table(lines):
    stats = []
    pair = []

    for line in lines:
        if is_reference_line(line):
            if pair:
                stats.append(compare_pairs(pair[0], pair[1]))
                pair = []
        elif is_data_line(line):
            pair.append(line)
            if len(pair) == 2:
                stats.append(compare_pairs(pair[0], pair[1]))
                pair = []
    
    # If there are remaining lines in the pair, they won't be processed as a complete pair
    return stats


def apply_highlight(input_file, output_file):
    content = read_file(input_file, False)
    content = highlight_table(content, 0.0750)
    write_file(content, output_file, False)


def apply_padding(input_file, output_file, num_decimals):
    content = read_file(input_file)
    content = pad_floats(content, num_decimals)
    write_file(content, output_file)


def apply_analysis(input_file):
    content = read_file(input_file, False)
    stats = process_table(content)
    increase = 0
    unchanged = 0
    decrease = 0
    for i, stat in enumerate(stats):
        increase += stat[0]
        decrease += stat[1]
        unchanged += stat[2]
    print(f'Increased: {increase} Decreased: {decrease} Unchanged: {unchanged}')

def process_latex_table(input_file, output_file, num_decimals, highlight, analysis):

    if highlight:
        apply_highlight(input_file, output_file)
        input_file = output_file

    if analysis:
        apply_analysis(input_file)
        
    apply_padding(input_file, output_file, num_decimals)

if __name__ == '__main__':

    parser = ArgumentParser(description='Process LaTeX file to highlight increased values.')
    parser.add_argument('input', type=str, help='Input LaTeX file path')
    parser.add_argument('output', type=str, help='Output LaTeX file path')
    parser.add_argument('decimal_cases', type=int, help='Number of decimal cases')
    parser.add_argument('--highlight', action='store_true', help='Highlight all values if provided')
    parser.add_argument('--analysis', action='store_true')

    args = parser.parse_args()
    process_latex_table(args.input, args.output, args.decimal_cases, args.highlight, args.analysis)