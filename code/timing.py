import numpy as np
from time import time
import math


def runtime(func, args=(), duration=5.0):
    """Time a function.

    Args:
        func (function): The function that is timed.
        args (tuple): The arguments with which the function is called
        duration (float): Approximate duration of the timing process in seconds

    Returns:
        runtimes_dict (dict): Dictionary with the following keys:
            - average_runtime
            - median_runtime
            - runtimes
            - standard_deviation
            - repetitions
            - string

    """
    t_zero = core_timer(func, args, 1)[0]

    iter_guess = math.floor(duration / t_zero)

    if iter_guess >= 100:
        t_zero = np.median(core_timer(func, args, 10))
        num_iter = math.floor(duration / t_zero)
    else:
        num_iter = iter_guess

    runtimes = core_timer(func, args, 1)
    num_iter = math.floor(duration / runtimes[0])

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
    runtimes = []
    for i in range(num_iter):
        start = time()
        func(*args)
        stop = time()
        raw_time = stop - start
        start = time()
        stop = time(*())
        delta = stop - start
        corrected_time = raw_time - delta
        runtimes.append(corrected_time)
    return runtimes


def find_good_unit(time):
    prefixes = ['', 'milli', 'micro', 'nano', 'pico', 'femto', 'atto']
    idx = 0
    while time < 1:
        time *= 1000
        idx += 1

    time = np.round(time, 2)
    return 'The median runtime is {} {}seconds'.format(time, prefixes[idx])

x = np.arange(10000)
y = np.ones(10000)


def test_func(x, y):
    return x.dot(y)

rt = runtime(test_func, args=(x, y))
print(rt['string'])
