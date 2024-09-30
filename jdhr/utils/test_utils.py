from jdhr.utils.console_utils import *
from jdhr.utils.net_utils import setup_deterministic

@catch_throw
def my_tests(globals: dict = globals(), prefix: str = 'test', fix_random: bool = True):
    # Setup deterministic testing environment
    setup_deterministic(fix_random)

    # Extract testing functions
    tests = {name: func for name, func in globals.items() if name.startswith(prefix)}

    # Run tests
    pbar = tqdm(total=len(tests))
    for name, func in tests.items():
        pbar.desc = name
        pbar.refresh()

        func()
        log(f'{name}: {green("OK")}')

        pbar.update(n=1)
        pbar.refresh()
