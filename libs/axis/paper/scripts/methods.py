"""Public method identities; cross-method comparisons are always explicit."""
METHODS = {
    "nearest": ("nearest", "same_method_family"),
    "bilinear": ("bilinear", "same_method_family"),
    "bicubic": ("patch", "cross_method"),
    "patch": ("patch", "same_method_family"),
    "conservative_first_order": ("conservative_first_order", "same_method_family"),
    "conservative_second_order": ("conservative_second_order", "same_method_family"),
    "conservative_second_order_limited": ("conservative_second_order", "limited_vs_unlimited"),
}
def pairing(method):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    esmc, kind = METHODS[method]
    return {"axis_method": method, "esmc_method": esmc, "comparison_kind": kind}
