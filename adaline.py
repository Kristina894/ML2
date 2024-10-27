def count_lines():
    count = 0
    with open("adaline.txt", "r") as f:
        for _ in f:
            count += 1
    return count

input_data = []
target = []

n = count_lines()
with open("adaline.txt", "r") as file:
    for line in file:
        line = line.strip()
        if ';' in line:
            before, after = line.split(';')
            data_row = [float(num) for num in before.split()]
            target.append(int(after))
            input_data.append(data_row)

for i in range(len(input_data)):
    input_data[i] = [1] + input_data[i]

print("Target:", target)
print("Input Data:", input_data)

w = []
w0 = float(input("Enter weight for bias (w0): "))
w.append(w0)

for i in range(len(input_data[0]) - 1):
    weight = float(input(f"Enter weight for feature {i + 1}: "))
    w.append(weight)

theta = float(input("Enter theta: "))
alpha = float(input("Enter alpha: "))
prev_weights = w.copy()

epoch = 0

while True:
    print(f"Epoch {epoch + 1}:")
    squared_sum = 0  
    for i in range(n):
        Ival = sum(w[j] * input_data[i][j] for j in range(len(w)))
        print("I: ", Ival)

        diff = target[i] - Ival
        squared_sum += diff ** 2
        
        if diff != 0:  
            for k in range(len(w)):
                if k == 0:
                    w[k] += alpha * diff
                else:
                    w[k] += alpha * diff * input_data[i][k]
    print(squared_sum)
    if squared_sum < theta:
        break
    epoch += 1
print("Final Weights:", w)
