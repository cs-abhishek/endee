import time
import numpy as np
from database_manager import DatabaseManager

db = DatabaseManager()
# Create a dummy vector for testing
test_vector = np.random.rand(384).tolist()

print("--- Starting Endee Performance Benchmark ---")

latencies = []
for i in range(10):
    start = time.perf_counter()
    db.search_vector(test_vector, limit=10)
    end = time.perf_counter()
    latencies.append((end - start) * 1000) # Convert to ms

avg_latency = sum(latencies) / len(latencies)
print(f"Average Endee Search Latency: {avg_latency:.2f} ms")
print(f"P95 Latency: {sorted(latencies)[-1]:.2f} ms")