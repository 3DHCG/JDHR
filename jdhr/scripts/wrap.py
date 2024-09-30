import os
import sys
import subprocess

from jdhr.utils.console_utils import *

import jittor as jt

JDHR = 'python -q -X faulthandler jdhr/scripts/main.py'


def configurable_entrypoint(SEPERATION='--', LAUNCHER='', JDHR=JDHR,
                            default_launcher_args=[],
                            extra_launcher_args=[],
                            default_jdhr_args=[],
                            extra_jdhr_args=[],
                            ):
    # Prepare for args
    args = sys.argv
    if SEPERATION in args:
        launcher_args = args[1:args.index(SEPERATION)]
        jdhr_args = args[args.index(SEPERATION) + 1:]
    else:
        launcher_args = default_launcher_args  # no extra arguments for torchrun (auto communimation, all available gpus)
        jdhr_args = args[1:] if len(args[1:]) else default_jdhr_args
    launcher_args += extra_launcher_args
    jdhr_args += extra_jdhr_args

    # Prepare for invokation
    args = []
    args.append(LAUNCHER)
    if launcher_args: args.append(' '.join(launcher_args))
    args.append(JDHR)
    if jdhr_args: args.append(' '.join(jdhr_args))

    # The actual invokation
    subprocess.call(' '.join(args), shell=True)


def dist_entrypoint():
    # Distribuated training
    configurable_entrypoint(LAUNCHER='torchrun', JDHR='jdhr/scripts/main.py', default_launcher_args=['--nproc_per_node', 'auto'], extra_jdhr_args=['distributed=True'])


def prof_entrypoint():
    # Profiling
    configurable_entrypoint(extra_jdhr_args=['profiler_cfg.enabled=True'])


def test_entrypoint():
    configurable_entrypoint(JDHR=JDHR + ' ' + '-t test')


def train_entrypoint():
    configurable_entrypoint(JDHR=JDHR + ' ' + '-t train')


def main_entrypoint():
    configurable_entrypoint()


def gui_entrypoint():
    # Directly run GUI without external requirements
    if '-c' not in sys.argv:
        sys.argv.insert(1, '-c')
        sys.argv.insert(2, 'configs/specs/gui.yaml')
    # else:
    #     cfg_idx = sys.argv.index('-c') + 1
    #     sys.argv[cfg_idx] = 'configs/base.yaml,' + sys.argv[cfg_idx]

    configurable_entrypoint(JDHR=JDHR + ' ' + '-t gui')


def ws_entrypoint():
    # Directly run GUI without external requirements
    if '-c' not in sys.argv:
        sys.argv.insert(1, '-c')
        sys.argv.insert(2, 'configs/base.yaml')
    # else:
    #     cfg_idx = sys.argv.index('-c') + 1
    #     sys.argv[cfg_idx] = 'configs/base.yaml,' + sys.argv[cfg_idx]

    args = sys.argv
    args = ['python -q -X faulthandler jdhr/scripts/client.py'] + args[1:]
    subprocess.call(' '.join(args), shell=True)
