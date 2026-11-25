import math

import numpy as np

from adtrain import rlpace


def _env(budget=10.0, scale=1.0):
    G = np.exp(rlpace.LG)
    spend = np.outer(np.full(rlpace.STEPS, 0.2 * scale), G)  # spend rises with gamma
    value = np.outer(np.full(rlpace.STEPS, 0.5 * scale), np.sqrt(G))
    return {"budget": budget, "spend": spend, "value": value}


def test_budget_is_a_hard_stop_and_mpc_spends_it():
    env = _env()
    plan = np.arange(1, rlpace.STEPS + 1) / rlpace.STEPS
    hi = rlpace.run(env, lambda t, s, st, e, p: math.log(2.0), plan)
    assert hi["spend"] <= env["budget"] + 1e-9 and hi["budget_out_at"] is not None
    mpc = rlpace.run(env, rlpace.mpc_policy(env), plan)
    assert abs(mpc["spend"] - env["budget"]) < 0.5 and mpc["tracking_error"] < 0.1


def test_pi_slows_down_when_ahead_of_plan():
    env = _env(scale=3.0)  # three times the traffic the plan expected
    plan = np.arange(1, rlpace.STEPS + 1) / rlpace.STEPS
    r = rlpace.run(env, rlpace.pi_policy, plan)
    assert r["trajectory"][48]["gamma"] < 0.6
