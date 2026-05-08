import argparse


def parse_labeled_path(
    spec: str,
    *,
    separators: tuple[str, ...] = ("=",),
) -> tuple[str, str]:
    """
    Parse a LABEL<sep>PATH command-line value.

    The default keeps older CLIs strict about LABEL=PATH. Callers that need to
    accept existing LABEL:PATH conventions can pass separators=("=", ":").
    """
    text = str(spec)
    positions = [
        (text.find(sep), sep) for sep in separators if sep and text.find(sep) >= 0
    ]
    if not positions:
        allowed = " or ".join(f"LABEL{sep}PATH" for sep in separators)
        raise ValueError(f"expected {allowed}, got {spec!r}")

    sep_idx, _sep = min(positions, key=lambda item: item[0])
    label = text[:sep_idx].strip()
    path = text[sep_idx + 1 :].strip()
    if not label or not path:
        raise ValueError(f"expected non-empty LABEL and PATH in {spec!r}")
    return label, path


def parse_labeled_path_arg(
    value: str,
    *,
    option_name: str = "--group",
    separators: tuple[str, ...] = ("=",),
) -> tuple[str, str]:
    try:
        return parse_labeled_path(value, separators=separators)
    except ValueError as exc:
        allowed = " or ".join(f"LABEL{sep}PATH" for sep in separators)
        raise argparse.ArgumentTypeError(
            f"{option_name} must be formatted as {allowed}."
        ) from exc


def parse_distances(distances_str):
    try:
        # Convert the string to a list of floats
        distances = list(map(float, distances_str.split(",")))

        # Check if the list is monotonically increasing
        if all(x < y for x, y in zip(distances, distances[1:])):
            return distances
        else:
            raise ValueError("The distances must be in monotonically increasing order.")
    except ValueError as e:
        print(f"Error: {e}")
        raise


def parse_training_selector(s: str) -> list[int]:
    """
    Parse a training selector string like:
      "1" or "1,3,5" or "1-3" or "1,3-5,8"
    Returns a list of 1-based training indices (sorted, unique).
    """
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []

    out: set[int] = set()
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a_str, b_str = token.split("-", 1)
            a = int(a_str.strip())
            b = int(b_str.strip())
            if a <= b:
                rng = range(a, b + 1)
            else:
                rng = range(b, a + 1)  # allow "5-3"
            for x in rng:
                out.add(int(x))
        else:
            out.add(int(token))

    return sorted(out)
