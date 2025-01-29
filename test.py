import matplotlib.pyplot as plt
import ast

with open('finetune_log_1.txt', 'r', encoding='utf-8') as file:
    loss_lines = []
    for line in file:
        line = line.strip()
        if line.startswith('{\'loss\''):
            try:
                loss = ast.literal_eval(line)['loss']
                loss_lines.append(loss)
            except Exception as e:
                print(f"Error parsing line: {line}")
                print(f"Error: {e}")

plt.plot(loss_lines)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss over Steps')
plt.show()
