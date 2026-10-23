from __future__ import annotations

import argparse

from integration.fault_injection import FaultConfig


def add_supervisor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--mpc-fault', choices=('none', 'solver_fail', 'nan', 'stale', 'exception', 'delay'), default='none')
    parser.add_argument('--fault-at', type=float, default=25.0, help='simulation seconds before temporary MPC fault')
    parser.add_argument('--fault-duration', type=float, default=2.0, help='fault window in simulation seconds')


def parse_fault(parser: argparse.ArgumentParser, args: argparse.Namespace) -> FaultConfig:
    if args.controller not in ('supervised', 'mpc_unprotected') and args.mpc_fault != 'none':
        parser.error('--mpc-fault requires --controller supervised or mpc_unprotected')
    duration = getattr(args, 'duration', None)
    if args.mpc_fault != 'none' and duration is not None and args.fault_at >= duration:
        parser.error('--fault-at %.2f s never fires within --duration %.2f s' % (args.fault_at, duration))
    try:
        return FaultConfig(args.mpc_fault, args.fault_at, args.fault_duration)
    except ValueError as error:
        parser.error(str(error))
