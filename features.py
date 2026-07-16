"""Chunk-level feature extraction for Poker44 bot detection.

Hands MUST be projected through poker44.validator.payload_view.prepare_hand_for_miner
before feature extraction, so training matches what the validator serves live.

All features are size-invariant (shares / means / stds), because benchmark chunk
groups hold 30-40 hands while live eval chunks hold ~100.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List

VISIBLE_BB = 0.02  # payload_view normalizes all money to sb=0.01 / bb=0.02
BUCKETS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 126.0)
ACTION_TYPES = ("check", "call", "bet", "raise", "fold")
STREETS = ("preflop", "flop", "turn", "river")


def _to_bb(money: Any) -> float:
    try:
        return float(money or 0.0) / VISIBLE_BB
    except (TypeError, ValueError):
        return 0.0


def _nearest_bucket_index(bb_value: float) -> int:
    return min(range(len(BUCKETS)), key=lambda i: abs(BUCKETS[i] - bb_value))


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            ent -= p * math.log(p)
    return ent


def _hand_stats(hand: Dict[str, Any]) -> Dict[str, Any]:
    metadata = hand.get("metadata") or {}
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    actions = hand.get("actions") or []
    hero_seat = int(metadata.get("hero_seat") or 0)

    # payload_view duplicates single-action hands to fill the window; collapse them.
    dup_window = False
    if len(actions) > 1 and all(a == actions[0] or (
        a.get("street") == actions[0].get("street")
        and a.get("actor_seat") == actions[0].get("actor_seat")
        and a.get("action_type") == actions[0].get("action_type")
        and a.get("normalized_amount_bb") == actions[0].get("normalized_amount_bb")
        and a.get("pot_before") == actions[0].get("pot_before")
    ) for a in actions[1:]):
        dup_window = True
        actions = actions[:1]

    stats: Dict[str, Any] = {
        "n_actions": len(actions),
        "dup_window": dup_window,
        "n_players": len(players),
        "n_streets": len(streets),
        "hero_seat": hero_seat,
        "max_seats": int(metadata.get("max_seats") or 0),
    }

    street_names = {str(s.get("street", "")).lower() for s in streets}
    for name in ("flop", "turn", "river", "showdown"):
        stats[f"reached_{name}"] = 1.0 if name in street_names else 0.0

    stacks_bb = [_to_bb(p.get("starting_stack")) for p in players]
    stats["stacks_bb"] = stacks_bb
    hero_stack = 0.0
    for p in players:
        if int(p.get("seat") or 0) == hero_seat:
            hero_stack = _to_bb(p.get("starting_stack"))
            break
    stats["hero_stack_bb"] = hero_stack

    type_counts = Counter()
    hero_type_counts = Counter()
    street_counts = Counter()
    hero_street_counts = Counter()
    actor_counts = Counter()
    amounts = []
    hero_amounts = []
    raise_to_vals = []
    call_to_vals = []
    pots_before = []
    pots_after = []

    for action in actions:
        a_type = str(action.get("action_type", ""))
        a_street = str(action.get("street", "")).lower()
        actor = int(action.get("actor_seat") or 0)
        amount_bb = float(action.get("normalized_amount_bb") or 0.0)
        is_hero = actor == hero_seat and hero_seat > 0

        type_counts[a_type] += 1
        street_counts[a_street] += 1
        actor_counts[actor] += 1
        if is_hero:
            hero_type_counts[a_type] += 1
            hero_street_counts[a_street] += 1
            if amount_bb > 0:
                hero_amounts.append(amount_bb)
        if amount_bb > 0:
            amounts.append(amount_bb)
        if action.get("raise_to"):
            raise_to_vals.append(_to_bb(action.get("raise_to")))
        if action.get("call_to"):
            call_to_vals.append(_to_bb(action.get("call_to")))
        pots_before.append(_to_bb(action.get("pot_before")))
        pots_after.append(_to_bb(action.get("pot_after")))

    stats.update(
        type_counts=type_counts,
        hero_type_counts=hero_type_counts,
        street_counts=street_counts,
        hero_street_counts=hero_street_counts,
        actor_counts=actor_counts,
        amounts=amounts,
        hero_amounts=hero_amounts,
        raise_to_vals=raise_to_vals,
        call_to_vals=call_to_vals,
        pots_before=pots_before,
        pots_after=pots_after,
        first_action=str(actions[0].get("action_type", "")) if actions else "",
        last_action=str(actions[-1].get("action_type", "")) if actions else "",
        hero_acted=1.0 if sum(hero_type_counts.values()) > 0 else 0.0,
        pot_final_bb=pots_after[-1] if pots_after else 0.0,
        pot_growth_bb=(pots_after[-1] - pots_before[0]) if pots_after else 0.0,
    )
    return stats


def extract_chunk_features(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """One flat feature dict for a chunk group of projected hands."""
    per_hand = [_hand_stats(h) for h in hands if isinstance(h, dict)]
    if not per_hand:
        return {}

    feats: Dict[str, float] = {}
    n = len(per_hand)

    # --- global action-type shares (table and hero) ---
    total_types = Counter()
    hero_types = Counter()
    total_streets = Counter()
    hero_streets = Counter()
    for hs in per_hand:
        total_types.update(hs["type_counts"])
        hero_types.update(hs["hero_type_counts"])
        total_streets.update(hs["street_counts"])
        hero_streets.update(hs["hero_street_counts"])

    total_actions = max(1, sum(total_types.values()))
    hero_actions = max(1, sum(hero_types.values()))
    for t in ACTION_TYPES:
        feats[f"share_{t}"] = total_types.get(t, 0) / total_actions
        feats[f"hero_share_{t}"] = hero_types.get(t, 0) / hero_actions
    aggressive = total_types.get("bet", 0) + total_types.get("raise", 0)
    passive = total_types.get("call", 0) + total_types.get("check", 0)
    feats["aggression_ratio"] = aggressive / max(1, passive)
    hero_aggr = hero_types.get("bet", 0) + hero_types.get("raise", 0)
    hero_pass = hero_types.get("call", 0) + hero_types.get("check", 0)
    feats["hero_aggression_ratio"] = hero_aggr / max(1, hero_pass)

    for s in STREETS:
        feats[f"{s}_action_share"] = total_streets.get(s, 0) / total_actions
        feats[f"hero_{s}_action_share"] = hero_streets.get(s, 0) / hero_actions
    feats["street_entropy"] = _entropy(total_streets)
    feats["action_type_entropy"] = _entropy(total_types)
    feats["hero_action_type_entropy"] = _entropy(hero_types)

    # --- amounts (bet sizing) ---
    all_amounts = [a for hs in per_hand for a in hs["amounts"]]
    hero_amounts = [a for hs in per_hand for a in hs["hero_amounts"]]
    for prefix, values in (("", all_amounts), ("hero_", hero_amounts)):
        feats[f"{prefix}mean_normalized_amount_bb"] = _mean(values)
        feats[f"{prefix}std_normalized_amount_bb"] = _std(values)
        feats[f"{prefix}max_amount_bb"] = max(values) if values else 0.0
        feats[f"{prefix}money_action_rate"] = len(values) / max(1, total_actions if not prefix else hero_actions)
    # bucket histogram (snap the deterministic noise back to canonical buckets)
    bucket_counts = Counter(_nearest_bucket_index(a) for a in all_amounts)
    for i in range(len(BUCKETS)):
        feats[f"amount_bucket_{i}"] = bucket_counts.get(i, 0) / max(1, len(all_amounts))
    hero_bucket_counts = Counter(_nearest_bucket_index(a) for a in hero_amounts)
    feats["hero_amount_bucket_entropy"] = _entropy(hero_bucket_counts)
    feats["amount_bucket_entropy"] = _entropy(bucket_counts)

    raise_to_all = [v for hs in per_hand for v in hs["raise_to_vals"]]
    call_to_all = [v for hs in per_hand for v in hs["call_to_vals"]]
    feats["raise_to_rate"] = len(raise_to_all) / max(1, total_actions)
    feats["call_to_rate"] = len(call_to_all) / max(1, total_actions)
    feats["mean_raise_to_bb"] = _mean(raise_to_all)
    feats["std_raise_to_bb"] = _std(raise_to_all)

    # --- pots ---
    pots_before_all = [v for hs in per_hand for v in hs["pots_before"]]
    feats["mean_pot_before"] = _mean(pots_before_all)
    feats["std_pot_before"] = _std(pots_before_all)
    finals = [hs["pot_final_bb"] for hs in per_hand]
    growth = [hs["pot_growth_bb"] for hs in per_hand]
    feats["mean_final_pot_bb"] = _mean(finals)
    feats["std_final_pot_bb"] = _std(finals)
    feats["mean_pot_growth_bb"] = _mean(growth)

    # --- per-hand shape distributions ---
    for key in ("n_actions", "n_streets", "n_players"):
        values = [float(hs[key]) for hs in per_hand]
        feats[f"mean_{key}"] = _mean(values)
        feats[f"std_{key}"] = _std(values)
    n_actions_counts = Counter(hs["n_actions"] for hs in per_hand)
    for k in (1, 2, 3, 4, 5, 6, 7, 8):
        feats[f"share_hands_{k}_actions"] = n_actions_counts.get(k, 0) / n
    feats["share_dup_window"] = sum(1 for hs in per_hand if hs["dup_window"]) / n

    for name in ("flop", "turn", "river", "showdown"):
        feats[f"share_reached_{name}"] = _mean([hs[f"reached_{name}"] for hs in per_hand])

    # --- stacks ---
    stacks_all = [s for hs in per_hand for s in hs["stacks_bb"] if s > 0]
    hero_stacks = [hs["hero_stack_bb"] for hs in per_hand if hs["hero_stack_bb"] > 0]
    feats["mean_starting_stack"] = _mean(stacks_all)
    feats["std_starting_stack"] = _std(stacks_all)
    feats["stack_cv"] = feats["std_starting_stack"] / max(1e-9, feats["mean_starting_stack"])
    feats["mean_hero_stack_bb"] = _mean(hero_stacks)
    feats["std_hero_stack_bb"] = _std(hero_stacks)
    feats["share_round_stacks"] = (
        sum(1 for s in stacks_all if abs(s - round(s / 50.0) * 50.0) < 0.6 and s > 0) / max(1, len(stacks_all))
    )

    # --- hero engagement / consistency (bots behave more uniformly hand-to-hand) ---
    feats["hero_acted_share"] = _mean([hs["hero_acted"] for hs in per_hand])
    hero_fold_flags = []
    hero_aggr_flags = []
    hero_n_actions = []
    for hs in per_hand:
        h_total = sum(hs["hero_type_counts"].values())
        hero_n_actions.append(float(h_total))
        if h_total > 0:
            hero_fold_flags.append(1.0 if hs["hero_type_counts"].get("fold", 0) > 0 else 0.0)
            h_aggr = hs["hero_type_counts"].get("bet", 0) + hs["hero_type_counts"].get("raise", 0)
            hero_aggr_flags.append(1.0 if h_aggr > 0 else 0.0)
    feats["hero_fold_hand_share"] = _mean(hero_fold_flags)
    feats["hero_fold_hand_std"] = _std(hero_fold_flags)
    feats["hero_aggr_hand_share"] = _mean(hero_aggr_flags)
    feats["hero_aggr_hand_std"] = _std(hero_aggr_flags)
    feats["mean_hero_actions_per_hand"] = _mean(hero_n_actions)
    feats["std_hero_actions_per_hand"] = _std(hero_n_actions)

    # --- table composition ---
    first_actions = Counter(hs["first_action"] for hs in per_hand)
    last_actions = Counter(hs["last_action"] for hs in per_hand)
    for t in ACTION_TYPES:
        feats[f"share_first_{t}"] = first_actions.get(t, 0) / n
        feats[f"share_last_{t}"] = last_actions.get(t, 0) / n
    hero_seats = Counter(hs["hero_seat"] for hs in per_hand)
    feats["hero_seat_entropy"] = _entropy(hero_seats)
    feats["mean_max_seats"] = _mean([float(hs["max_seats"]) for hs in per_hand])
    actor_conc = []
    for hs in per_hand:
        total = sum(hs["actor_counts"].values())
        if total > 0:
            actor_conc.append(max(hs["actor_counts"].values()) / total)
    feats["mean_actor_concentration"] = _mean(actor_conc)

    return feats
