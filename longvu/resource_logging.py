import logging
import time
import torch
import inspect
import os
from pathlib import Path
from typing import List, Dict

def get_cuda_devices() -> List[torch.device]:
    if not torch.cuda.is_available():
        return []  # Return an empty list if CUDA is not available
    return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

# Get all Python files in the project directory
def get_project_py_files():
    current_dir = Path.cwd()  # Get the current working directory
    project_py_files = set()
    for root, _, files in os.walk(current_dir):
        for file in files:
            if "resource_logging" not in file and file.endswith(".py"):  # Collect all .py files except resource_logging.py
                project_py_files.add(Path(root) / file)
    return project_py_files

def measure_resource_usage(devices: List[torch.device] = get_cuda_devices()):
    # Validate the device and collect project Python files
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this system.")
    for device in devices:
        if "cuda" not in device.type:
            raise ValueError("The device must be a CUDA device.")
    
    project_py_files = set(get_project_py_files())  # Cache project Python files

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                # Measure start time and memory
                start_time = time.time()
                start_allocated: Dict[torch.device, int] = {}
                start_reserved: Dict[torch.device, int] = {}
                for device in devices:
                    start_allocated[device] = torch.cuda.memory_allocated(device)
                    start_reserved[device] = torch.cuda.memory_reserved(device)

                # Execute the function
                result = func(*args, **kwargs)

                # Measure end time and memory
                end_time = time.time()
                end_allocated: Dict[torch.device, int] = {}
                end_reserved: Dict[torch.device, int] = {}
                for device in devices:
                    end_allocated[device] = torch.cuda.memory_allocated(device)
                    end_reserved[device] = torch.cuda.memory_reserved(device)
                peak_allocated: Dict[torch.device, int] = {}
                for device in devices:
                    peak_allocated[device] = torch.cuda.max_memory_allocated(device)

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

                logging.info(f'File: {caller_filename}, Line: {caller_lineno}')
                logging.info(f'Time: {end_time - start_time:.2f} seconds')
                for device in devices:
                    logging.info(f'Device: {device}')
                    logging.info(f'Allocated before: {start_allocated[device]/1e6:.2f} MB')
                    logging.info(f'Allocated after:  {end_allocated[device]/1e6:.2f} MB')
                    logging.info(f'Net allocated change:  {(end_allocated[device] - start_allocated[device])/1e6:.2f} MB')
                    logging.info(f'Reserved before:  {start_reserved[device]/1e6:.2f} MB')
                    logging.info(f'Reserved after:   {end_reserved[device]/1e6:.2f} MB')
                    logging.info(f'Net reserved change:   {(end_reserved[device] - start_reserved[device])/1e6:.2f} MB')
                    logging.info(f'Peak allocated:         {peak_allocated[device]/1e6:.2f} MB')

                return result
            except Exception as e:
                logging.error(f"Error in measure_cuda_usage for {func.__name__}: {e}")
                raise  # Re-raise the exception after logging
        return wrapper
    return decorator

class MeasureResourceUsage:
    def __init__(self, devices: List[torch.device] = get_cuda_devices()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this system.")
        for device in devices:
            if "cuda" not in device.type:
                raise ValueError("The device must be a CUDA device.")
        
        self.devices = devices
        self.project_py_files = set(get_project_py_files())  # Use set for faster lookups

    def __enter__(self):
        self.start_time = time.time()
        for device in self.devices:
            self.start_allocated[device] = torch.cuda.memory_allocated(device)
            self.start_reserved[device] = torch.cuda.memory_reserved(device)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            # Measure end time and memory
            end_time = time.time()
            end_allocated: Dict[torch.device, int] = {}
            end_reserved: Dict[torch.device, int] = {}
            peak_allocated: Dict[torch.device, int] = {}
            for device in self.devices:
                end_allocated[device] = torch.cuda.memory_allocated(device)
                end_reserved[device] = torch.cuda.memory_reserved(device)
                peak_allocated[device] = torch.cuda.max_memory_allocated(device)

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
            logging.info(f'File: {caller_filename}, Line: {caller_lineno}')
            logging.info(f'Time: {end_time - self.start_time:.2f} seconds')
            for device in self.devices:
                logging.info(f'Device: {device}')
                logging.info(f'Allocated before block: {self.start_allocated[device]/1e6:.2f} MB')
                logging.info(f'Allocated after block:  {end_allocated[device]/1e6:.2f} MB')
                logging.info(f'Net allocated change:  {(end_allocated[device] - self.start_allocated[device])/1e6:.2f} MB')
                logging.info(f'Reserved before block:  {self.start_reserved[device]/1e6:.2f} MB')
                logging.info(f'Reserved after block:   {end_reserved[device]/1e6:.2f} MB')
                logging.info(f'Net reserved change:  {(end_reserved[device] - self.start_reserved[device])/1e6:.2f} MB')
                logging.info(f'Peak allocated:         {peak_allocated[device]/1e6:.2f} MB')
        except Exception as e:
            logging.error(f"Error in MeasureResourceUsage: {e}")