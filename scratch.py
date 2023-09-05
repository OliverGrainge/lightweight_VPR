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


def benchmark_throughput(model, input_shape=(3, 224, 224)):
  # Make sure model is on GPU
  model = model.cuda()

  # Create dummy data
  batch_size = compute_batch_size(model, input_shape=input_shape)
  torch.cuda.empty_cache()
  input_tensor = torch.randn(batch_size, 3, 224, 224).cuda()

  # Warm-up
  for _ in range(10):
      with torch.no_grad():
        output = model(input_tensor)

  # Measure throughput using CUDA events
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  start_event.record()

  num_batches = 100
  for _ in range(num_batches):
      output = model(input_tensor)

  end_event.record()
  torch.cuda.synchronize()

  elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
  elapsed_time = elapsed_time / 1000  # Convert to seconds

  throughput = (batch_size * num_batches) / elapsed_time
  return throughput


def compute_batch_size(model, input_shape=(3, 224, 224), delta=10):
  model = model.cuda()  # Ensure the model is on GPU

  # Starting batch size
  batch_size = 1


  # Loop to find maximum runnable batch size
  while True:
      try:
          print(batch_size)
          input_tensor = torch.randn((batch_size,) + input_shape).cuda()
          
          # Try a forward pass
          with torch.no_grad():
            output = model(input_tensor)
          
          # Clear the GPU memory
          del input_tensor, output  
          torch.cuda.empty_cache()
          
          # If successful, double the batch size
          batch_size *= 2 
          
      except RuntimeError as e:
          if 'out of memory' in str(e):
              del model
              torch.cuda.empty_cache()
              return int(batch_size / 2)
              break
          else:
              raise e
  