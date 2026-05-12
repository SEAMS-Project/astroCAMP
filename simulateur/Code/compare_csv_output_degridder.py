import csv
import math

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        lines = [line for line in reader if line]
        count = int(lines[0][0])  # first line: number of rows
        data = [[float(val) for val in line if val] for line in lines[1:]]
        return count, data

def compare_csv(file1, file2, tolerance=0.001):
    count1, data1 = read_csv(file1)
    count2, data2 = read_csv(file2)

    if count1 != count2:
        print(f"Different number of lines: {count1} vs {count2}")
        return

    diffs = []
    for i, (row1, row2) in enumerate(zip(data1, data2), start=2):
        for j, (val1, val2) in enumerate(zip(row1, row2)):
            if not math.isclose(val1, val2, abs_tol=tolerance):
                diffs.append((i, j, val1, val2))

    if not diffs:
        print("✅ The files are equivalent within the given tolerance.")
    else:
        print(f"❌ Differences found at {len(diffs)} positions:")
        for i, j, v1, v2 in diffs:
            #if(j!=4):
            print(f"Line {i}, Column {j+1}: {v1} vs {v2}")

compare_csv('output/output_degridder.csv', 'output/ref2560.csv', 0.0)


