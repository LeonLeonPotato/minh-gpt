import json
import matplotlib.pyplot as plt
import numpy as np

data = []
with open('results.jsonl') as f:
    for l in f.readlines():
        data.append(json.loads(l)['loss'])

x = np.linspace(1, len(data), len(data))
A, B = np.polyfit(np.log(x), data, 1)
y = A * np.log(np.linspace(1, len(data) * 5, len(data) * 5)) + B

plt.plot(y)
plt.plot(np.convolve(data, np.ones(10), 'valid') / 10)
plt.plot(data)
plt.show()