
import pickle 
import numpy as np


with open("recall.pkl", "rb") as f:
    recall = pickle.load(f)
    
with open("latency.pkl", "rb") as f:
    latency = pickle.load(f)

with open("memory.pkl", "rb") as f:
    memory = pickle.load(f)

with open("configs.pkl", "rb") as f:
    configs = pickle.load(f)

#print(configs[np.argmin(memory)])

#print(configs[np.argmin(latency)])

#print(configs[np.argmax(np.array(recall)[:, 0])])




def compute_scores(weights, recall, latency, memory):
    benchmark_recall = 69.26
    benchmark_latency = 2.778
    benchmark_memory = 12841612

    recall = np.array(recall)[:, 0]
    latency = np.array(latency)
    memory = np.array(memory)

    scores = []
    for i in range(len(recall)):
        memory_change = ((memory[i]/benchmark_memory) - 1) * 100
        latency_change = ((latency[i]/benchmark_latency) - 1) * 100
        recall_change = ((recall[i]/benchmark_recall) - 1) * 100
        loss = weights[2] * memory_change + weights[1]* latency_change - weights[0] * recall_change
        scores.append(loss)

    idx = np.argmin(scores)
    return idx


IMPORTANCE_WEIGHTS = [50., 1.0, 0.16] 
IMPORTANCE_WEIGHTS = [1., 1., 1.]
idx = compute_scores(IMPORTANCE_WEIGHTS, recall, latency, memory)
print(configs[idx])

print(np.array(recall)[:, 0])