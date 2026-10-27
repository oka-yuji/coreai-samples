import math


def rating_identity_matches(left, right):
    if type(left) in (int, float) and type(right) in (int, float):
        return all(type(value) is not float or math.isfinite(value) for value in (left, right)) and left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(rating_identity_matches(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(rating_identity_matches(a, b) for a, b in zip(left, right, strict=True))
    return type(left) in (str, bool, type(None)) and left == right
