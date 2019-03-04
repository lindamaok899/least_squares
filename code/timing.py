import numpy as np
import time
import math


def runtime(func, args=(), duration=1.0):
    """Time a function.

    Args:
        func (function): The function that is timed.
        args (tuple): The arguments with which the function is called
        duration (float): Approximate duration of the timing process in seconds

    Returns:
        runtimes_dict (dict): Dictionary with the following keys:
            - average_runtime
            - median_runtimes
            - runtimes
            - standard_deviation
            - repetitions
            - string

    """
    t_zero = core_timer(func, args, 2)[1]
    small_time = time.get_clock_info('perf_counter').resolution
    iter_guess = max(1, math.floor(duration / max(t_zero, small_time)))

    if iter_guess >= 100:
        t_zero = np.median(core_timer(func, args, 10))
        num_iter = max(1, math.floor(duration / max(t_zero, small_time)))
    else:
        num_iter = iter_guess
    runtimes = core_timer(func, args, num_iter)
    avg_runtime = np.mean(runtimes)
    median_runtime = np.median(runtimes)

    runtime_dict = {
        'average_runtime': avg_runtime,
        'median_runtime': np.median(runtimes),
        'runtimes': runtimes,
        'standard_deviation': np.std(runtimes),
        'repetitions': num_iter,
        'string': find_good_unit(median_runtime)
    }
    return runtime_dict


def core_timer(func, args=(), num_iter=1):
    """Gets the initial runtime data for functions.
        Args:
            func (function): The function that is timed.
            args (tuple): The arguments with which the function is called.
            num_iter (float): number of iterations, default is one.
            
        Returns:
            runtimes (float): runtime unit for each function passed in.           
    
    """
    
    runtimes = []
    for i in range(num_iter):
        start = time.perf_counter()
        func(*args)
        stop = time.perf_counter()
        raw_time = stop - start
        start = time.perf_counter()
        stop = time.perf_counter()
        delta = stop - start
        corrected_time = raw_time - delta
        runtimes.append(corrected_time)
    return runtimes


def find_good_unit(time):
    """Finds appropriate representational units of time.
    Args:
        time (float): time for which a unit needs to be assigned.
    
    Returns: 
        time (float): time adjusted to appropriate unit.
    
    """
    
    prefixes = ['', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto']
    idx = 0
    while time < 1:
        time *= 1000
        idx += 1

    time = np.round(time, 2)
    return 'The median runtime is {} {}seconds'.format(time, prefixes[idx])
