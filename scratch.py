import torch 
import numpy as np


def benchmark(model, input_size=(3, 224, 224), batch_size=(12,), repetitions=300):
  input_tensor = torch.randn(batch_size + input_size, dtype=torch.float32).to("cuda")
  model.to("cuda")
  print(input_tensor.shape)
  for _ in range(10):
    output = model(input_tensor)

  # Time measurement using CUDA events
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  latencies = []
  for _ in range(repetitions):
      start_event.record()
      output = model(input_tensor)
      end_event.record()
      torch.cuda.synchronize()
      latencies.append(start_event.elapsed_time(end_event))

  average_latency = np.mean(latencies)
  std_latency = np.std(latencies)
  return average_latency, std_latency
  