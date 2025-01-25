import logging
import time
import torch
import inspect
import os
from pathlib import Path

# Get all Python files in the project directory
def get_project_py_files():
    current_dir = Path.cwd()  # Get the current working directory
    project_py_files = set()
    for root, _, files in os.walk(current_dir):
        for file in files:
            if "resource_logging" not in file and file.endswith(".py"):  # Collect all .py files except resource_logging.py
                project_py_files.add(Path(root) / file)
    return project_py_files

def measure_resource_usage(device=torch.device("cuda:0")):
    # Validate the device and collect project Python files
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")
    if device.type != "cuda":
        raise ValueError("The device must be a CUDA device.")
    
    project_py_files = set(get_project_py_files())  # Cache project Python files

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                # Measure start time and memory
                start_time = time.time()
                start_allocated = torch.cuda.memory_allocated(device)
                start_reserved = torch.cuda.memory_reserved(device)

                # Execute the function
                result = func(*args, **kwargs)

                # Measure end time and memory
                end_time = time.time()
                end_allocated = torch.cuda.memory_allocated(device)
                end_reserved = torch.cuda.memory_reserved(device)
                peak_allocated = torch.cuda.max_memory_allocated(device)

                # Identify the caller frame within the project directory
                caller_filename = "Unknown"
                caller_lineno = "Unknown"
                stack = inspect.stack()
                for frame in stack:
                    try:
                        frame_file = Path(frame.filename).resolve()
                        if frame_file in project_py_files:  # Check if the file is part of the project
                            caller_filename = frame_file
                            caller_lineno = frame.lineno
                            break
                    except Exception:
                        continue  # Skip problematic frames

                # Log memory and time usage with caller info
                logging.info(
                    f"\nFile: {caller_filename}, Line: {caller_lineno}\n"
                    f"Allocated before: {start_allocated/1e6:.2f} MB\n"
                    f"Allocated after:  {end_allocated/1e6:.2f} MB\n"
                    f"Net allocated change:  {(end_allocated - start_allocated)/1e6:.2f} MB\n"
                    f"Reserved before:  {start_reserved/1e6:.2f} MB\n"
                    f"Reserved after:   {end_reserved/1e6:.2f} MB\n"
                    f"Net reserved change:   {(end_reserved - start_reserved)/1e6:.2f} MB\n"
                    f"Peak allocated:         {peak_allocated/1e6:.2f} MB\n"
                    f"Runtime: {end_time - start_time:.2f} seconds"
                )

                return result
            except Exception as e:
                logging.error(f"Error in measure_cuda_usage for {func.__name__}: {e}")
                raise  # Re-raise the exception after logging
        return wrapper
    return decorator

class MeasureResourceUsage:
    def __init__(self, device=torch.device("cuda:0")):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system.")
        if device.type != "cuda":
            raise ValueError("The device must be a CUDA device.")
        
        self.device = device
        self.project_py_files = set(get_project_py_files())  # Use set for faster lookups

    def __enter__(self):
        self.start_time = time.time()
        self.start_allocated = torch.cuda.memory_allocated(self.device)
        self.start_reserved = torch.cuda.memory_reserved(self.device)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            # Measure end time and memory
            end_time = time.time()
            end_allocated = torch.cuda.memory_allocated(self.device)
            end_reserved = torch.cuda.memory_reserved(self.device)
            peak_allocated = torch.cuda.max_memory_allocated(self.device)

            # Identify the caller frame within the project directory
            stack = inspect.stack()
            caller_filename = "Unknown"
            caller_lineno = "Unknown"
            for frame in stack:
                try:
                    frame_file = Path(frame.filename).resolve()
                    if frame_file in self.project_py_files:  # Check if the file is part of the project
                        caller_filename = frame_file
                        caller_lineno = frame.lineno
                        break
                except Exception:
                    continue  # Skip problematic frames

            # Log memory and time usage with caller info
            logging.info(
                f"\nFile: {caller_filename}, Line: {caller_lineno}\n"
                f"Allocated before block: {self.start_allocated/1e6:.2f} MB\n"
                f"Allocated after block:  {end_allocated/1e6:.2f} MB\n"
                f"Net allocated change:  {(end_allocated - self.start_allocated)/1e6:.2f} MB\n"
                f"Reserved before block:  {self.start_reserved/1e6:.2f} MB\n"
                f"Reserved after block:   {end_reserved/1e6:.2f} MB\n"
                f"Net reserved change:  {(end_reserved - self.start_reserved)/1e6:.2f} MB\n"
                f"Peak allocated:         {peak_allocated/1e6:.2f} MB\n"
                f"Runtime: {end_time - self.start_time:.2f} seconds"
            )
        except Exception as e:
            logging.error(f"Error in MeasureResourceUsage: {e}")