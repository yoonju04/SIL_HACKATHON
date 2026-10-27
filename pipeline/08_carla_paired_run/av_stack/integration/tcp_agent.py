from __future__ import annotations

from integration.tcp_predictor import TCPOnlySensorAgent
from integration.tfpp_agent import TFPPAgent


def get_entry_point() -> str:
    return 'TCPAgent'


class TCPAgent(TCPOnlySensorAgent):
    setup = TFPPAgent.setup
    run_step = TFPPAgent.run_step

    def destroy(self) -> None:
        TFPPAgent.destroy(self)
        TCPOnlySensorAgent.destroy(self)
