"""adsim: the simulated mobile ad market behind *Machine Learning in Mobile
AdTech*.

Typical use:

    from adsim import preset, simulate
    result = simulate(preset("crawl", seed=7))
    result.write("out/crawl")
"""
from .config import SimConfig, preset
from .run import SimResult, simulate

__all__ = ["SimConfig", "SimResult", "preset", "simulate"]
__version__ = "0.3.0"
