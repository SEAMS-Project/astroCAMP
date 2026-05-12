import csv
import math

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = [line for line in reader if line]
        count = sum(1 for _ in lines)
        data = [[float(val) for val in line] for line in lines]
        return count, data

def compare_csv(file1, file2, tolerance=0.001):
    count1, data1 = read_csv(file1)
    count2, data2 = read_csv(file2)

    if count1 != count2:
        print(f"Different number of lines: {count1} vs {count2}")
        return

    diffs = []
    for i, (row1, row2) in enumerate(zip(data1, data2), start=1):
        for j, (val1, val2) in enumerate(zip(row1, row2)):
            if not math.isclose(val1, val2, abs_tol=tolerance):
                diffs.append((i, j+1, val1, val2))

    if not diffs:
        print("✅ The files are equivalent within the given tolerance.")
    else:
        print(f"❌ Differences found at {len(diffs)} positions:")
        for i, j, v1, v2 in diffs:
            print(f"Line {i}, Column {j}: {v1} vs {v2}")

compare_csv('output/gridding32.csv', 'output/gridding.csv', 0.0)

