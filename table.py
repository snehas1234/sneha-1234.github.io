def pattern(sequence):
   
    common_diff = sequence[1]-sequence[0]
    for i in range(2, len(sequence)):
        if sequence[i]-sequence[i-1] != common_diff:
            break
    else:
        return f"Arithmetic progression, common difference: {common_diff}"

    
    common_ratio = sequence[1]/sequence[0]
    for i in range(2, len(sequence)):
        if sequence[i]/sequence[i-1] != common_ratio:
            break
    else:
        return f"Geometric progression, common ratio: {common_ratio}"

    
    for i in range(2, len(sequence)):
        if sequence[i] != sequence[i-1] + sequence[i-2]:
            break
    else:
        return "Fibonacci sequence"

    return "Pattern not recognized"



sequences = {
    1: [2, 4, 6, 8, 10],
    2: [3, 6, 12, 24, 48],
    3: [1, 1, 2, 3, 5, 8],
    4: [10, 20, 30, 55, 70],
    5: [100, 50, 25, 12.5, 6.25]
}


for index, seq in sequences.items():
    print(f"Index {index}: {seq} -> {pattern(seq)}")