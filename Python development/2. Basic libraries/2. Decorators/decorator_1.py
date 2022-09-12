# simple wrapper
def do(func):
    def wrapper():
        func()
        func()
    return wrapper

# with any number of arbitrary arguments

def do_twice(func):
    def wrapper_1(*args, **kwargs):
        func(*args, **kwargs)
        func(*args, **kwargs)
    return wrapper_1

# Returning Values From Decorated Functions

def do_twice(func):
    def wrapper_1(*args, **kwargs):
        func(*args, **kwargs)
        func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper_1

# Decorated knows it identity and doesn't claim it to be the wrapper inside the decorator function.

import functools

def do_twice_functools(func):
    @functools.wraps(func)
    def wrapper_1(*args, **kwargs):
        func(*args, **kwargs)
        func(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper_1