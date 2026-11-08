"""Bounded streaming helpers for raw nuScenes metadata tables."""

import ijson


def load_sample_data(path, raw_sample_tokens, explicit_tokens):
    """Retain selected raw keyframes and explicitly referenced enriched rows.

    Unlike eager JSON loading, memory grows with retained keyframes rather than
    the full sample-data table. Both token collections are sets of strings.
    """
    selected = {}
    with open(path, "rb") as handle:
        for item in ijson.items(handle, "item", use_float=True):
            if (item["token"] in explicit_tokens or
                    (item.get("sample_token") in raw_sample_tokens and item.get("is_key_frame") is True)):
                selected[item["token"]] = item
    return selected
