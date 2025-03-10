import os
import time
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Union
import cv2
import numpy as np


class File:
    """A simple file wrapper similar to Sieve's File class."""

    def __init__(self, path=None, url=None):
        self.path = path
        self.url = url

        # If URL is provided but not path, download the file
        if url and not path:
            import requests
            import tempfile

            print(f"Downloading file from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Create a temporary file with the right extension
            ext = os.path.splitext(url.split('?')[0])[1]
            fd, self.path = tempfile.mkstemp(suffix=ext)
            os.close(fd)

            # Write the content to the file
            with open(self.path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded file to {self.path}")


class Image:
    """Simple Image metadata class."""

    def __init__(self, url=None):
        self.url = url


class Metadata:
    """Simple metadata class for functions and models."""

    def __init__(self, title=None, description=None, tags=None, code_url=None, image=None, readme=None):
        self.title = title
        self.description = description
        self.tags = tags or []
        self.code_url = code_url
        self.image = image
        self.readme = readme


class Future:
    """A simple Future class to mimic Sieve's async behavior."""

    # Thread pool to limit concurrent executions
    _thread_pool_lock = threading.RLock()
    _active_threads = 0
    _max_threads = 1  # Default maximum threads
    _queue = []
    _queue_condition = threading.Condition(_thread_pool_lock)

    @classmethod
    def set_max_threads(cls, max_threads):
        """Set the maximum number of concurrent threads."""
        with cls._thread_pool_lock:
            cls._max_threads = max_threads
            # Notify waiting threads in case the limit was increased
            cls._queue_condition.notify_all()

    @classmethod
    def get_max_threads(cls):
        """Get the current maximum thread limit."""
        with cls._thread_pool_lock:
            return cls._max_threads

    @classmethod
    def get_active_threads(cls):
        """Get the current number of active threads."""
        with cls._thread_pool_lock:
            return cls._active_threads

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result_value = None
        self.exception = None
        self.completed = False
        self.thread = None
        self._queued = False

        # Queue the task for execution
        self._queue_execution()

    def _queue_execution(self):
        with Future._thread_pool_lock:
            if Future._active_threads < Future._max_threads:
                # Start immediately if under the thread limit
                Future._active_threads += 1
                self.thread = threading.Thread(target=self._execute)
                self.thread.start()
            else:
                # Queue for later execution
                self._queued = True
                Future._queue.append(self)

    def _execute(self):
        try:
            self.result_value = self.func(*self.args, **self.kwargs)
        except Exception as e:
            self.exception = e
        finally:
            self.completed = True

            # Release the thread and start the next queued task if any
            with Future._thread_pool_lock:
                Future._active_threads -= 1
                if Future._queue:
                    next_task = Future._queue.pop(0)
                    next_task._queued = False
                    Future._active_threads += 1
                    next_task.thread = threading.Thread(
                        target=next_task._execute)
                    next_task.thread.start()
                # Notify any waiting threads
                Future._queue_condition.notify_all()

    def done(self):
        return self.completed

    def result(self):
        # If the task is still in the queue, wait for it to start
        if self._queued:
            with Future._queue_condition:
                while self._queued:
                    Future._queue_condition.wait()

        # Now wait for the thread to complete
        if self.thread:
            self.thread.join()

        if self.exception:
            raise self.exception
        return self.result_value


class gpu:
    """GPU specification class."""
    @staticmethod
    def T4(split=1):
        return {"type": "T4", "split": split}

    @staticmethod
    def L4(split=1):
        return {"type": "L4", "split": split}


class function:
    """Function decorator and manager."""
    _registry = {}
    _base_log_dir = "sieve_logs"
    _call_stack = []
    _function_counters = {}
    _current_execution_path = None
    _global_counter = 0  # Global counter for top-level functions

    def __init__(self, name=None, python_version=None, metadata=None, python_packages=None,
                 system_packages=None, run_commands=None):
        self.name = name
        self.python_version = python_version
        self.metadata = metadata
        self.python_packages = python_packages
        self.system_packages = system_packages
        self.run_commands = run_commands

    @staticmethod
    def _execute_with_logging(func, func_name, args, kwargs, via_method="direct"):
        """Common execution logic with logging for both direct calls and push() calls."""
        # Create base log directory if it doesn't exist
        if not os.path.exists(function._base_log_dir):
            os.makedirs(function._base_log_dir)

        # Determine if this is a top-level call or nested call
        is_top_level = len(function._call_stack) == 0

        # For top-level functions, use the global counter
        # For nested functions, use the parent's execution path as the counter key
        if is_top_level:
            function._global_counter += 1
            counter = function._global_counter
            counter_key = "global"
        else:
            # Use only the parent's execution path as the counter key
            counter_key = function._current_execution_path

            if counter_key not in function._function_counters:
                function._function_counters[counter_key] = 0
            function._function_counters[counter_key] += 1
            counter = function._function_counters[counter_key]

        # Create the execution ID for this specific function call with the new format
        execution_id = f"{counter:04d}_{func_name.replace('/', '_')}"

        # Create directory path and new execution path
        if is_top_level:
            log_dir = os.path.join(function._base_log_dir, execution_id)
            new_execution_path = execution_id
        else:
            log_dir = os.path.join(
                function._base_log_dir, function._current_execution_path, execution_id)
            new_execution_path = os.path.join(
                function._current_execution_path, execution_id)

        os.makedirs(log_dir, exist_ok=True)

        # Log parameters
        params_log_path = os.path.join(log_dir, "parameters.txt")
        with open(params_log_path, "w") as f:
            f.write(f"Function: {func_name}\n")
            f.write(f"Execution ID: {execution_id}\n")
            f.write(f"Counter: {counter}\n")
            f.write(f"Counter Key: {counter_key}\n")
            f.write(f"Call stack: {function._call_stack + [func_name]}\n")
            f.write(f"Args: {args}\n")
            f.write(f"Kwargs: {kwargs}\n")
            f.write(f"Called via: {via_method}\n")

        # Update call stack and execution path for nested calls
        previous_path = function._current_execution_path
        function._call_stack.append(func_name)
        function._current_execution_path = new_execution_path

        # Execute function and time it
        start_time = time.time()
        try:
            result = func(*args, **kwargs)

            # Check if result is a generator and handle it specially
            if hasattr(result, '__iter__') and hasattr(result, '__next__'):
                # For generators, we need to consume the generator to get accurate timing
                def timed_generator():
                    try:
                        for item in result:
                            yield item
                    finally:
                        # Log execution time after generator is fully consumed
                        end_time = time.time()
                        execution_time = end_time - start_time
                        time_log_path = os.path.join(
                            log_dir, "execution_time.txt")
                        with open(time_log_path, "w") as f:
                            f.write(
                                f"Execution time: {execution_time:.4f} seconds\n")

                        # Restore call stack and execution path after generator is consumed
                        function._call_stack.pop()
                        function._current_execution_path = previous_path

                # Return the wrapped generator without popping from the stack yet
                return timed_generator()
            else:
                # For regular functions, log time immediately
                end_time = time.time()
                execution_time = end_time - start_time
                time_log_path = os.path.join(log_dir, "execution_time.txt")
                with open(time_log_path, "w") as f:
                    f.write(f"Execution time: {execution_time:.4f} seconds\n")

                # Restore call stack and execution path for regular functions
                function._call_stack.pop()
                function._current_execution_path = previous_path

                return result

        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time

            # Log error and execution time
            error_log_path = os.path.join(log_dir, "error.txt")
            with open(error_log_path, "w") as f:
                f.write(f"Error: {str(e)}\n")
                f.write(
                    f"Execution time before error: {execution_time:.4f} seconds\n")

            # Restore call stack and execution path in case of error
            function._call_stack.pop()
            function._current_execution_path = previous_path

            raise

    def __call__(self, func):
        # Register the function
        func_name = self.name or func.__name__
        original_func = func

        # Create a wrapper that logs execution details
        def logged_func(*args, **kwargs):
            return function._execute_with_logging(original_func, func_name, args, kwargs)

        # Replace the original function with our logged version
        function._registry[func_name] = logged_func
        return logged_func

    @staticmethod
    def get(name):
        """Get a function by name."""
        if name not in function._registry:
            raise ValueError(f"Function {name} not found in registry")

        # Create a wrapper that mimics the Sieve push method
        class FunctionWrapper:
            def __init__(self, func):
                self.func = func

            def push(self, *args, **kwargs):
                # Create a wrapper that will perform the logging when executed by the Future
                def logged_push_execution(*args, **kwargs):
                    return function._execute_with_logging(self.func, name, args, kwargs, via_method="push")

                # Return a Future that will execute our logged function
                return Future(logged_push_execution, *args, **kwargs)

        return FunctionWrapper(function._registry[name])


class Model:
    """Model decorator and manager."""
    _registry = {}

    def __init__(self, name=None, gpu=None, python_packages=None, cuda_version=None,
                 system_packages=None, python_version=None, metadata=None, run_commands=None):
        self.name = name
        self.gpu = gpu
        self.python_packages = python_packages
        self.cuda_version = cuda_version
        self.system_packages = system_packages
        self.python_version = python_version
        self.metadata = metadata
        self.run_commands = run_commands

    def __call__(self, cls):
        # Register the model class
        name = self.name or cls.__name__
        Model._registry[name] = cls

        # Create an instance of the model
        instance = cls()
        if hasattr(instance, "__setup__"):
            print(f"Setting up model {name}...")
            instance.__setup__()

        # Create a wrapper that mimics the Sieve push method
        class ModelWrapper:
            def __init__(self, instance):
                self.instance = instance

            def push(self, *args, **kwargs):
                return Future(self.instance.__predict__, *args, **kwargs)

        function._registry[name] = instance.__predict__
        return cls

    @staticmethod
    def get(name):
        """Get a model by name."""
        if name not in Model._registry:
            raise ValueError(f"Model {name} not found in registry")

        # Create a wrapper that mimics the Sieve push method
        class ModelWrapper:
            def __init__(self, cls):
                self.instance = cls()
                if hasattr(self.instance, "__setup__"):
                    self.instance.__setup__()

            def push(self, *args, **kwargs):
                return Future(self.instance.__predict__, *args, **kwargs)

        return ModelWrapper(Model._registry[name])
