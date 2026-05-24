import typing as t


def normalize_backend_url(url: str) -> str:
    return (url or "").strip().rstrip("/")


def parse_backend_candidates(configured_url: str = "", candidates_csv: str = "") -> list[str]:
    urls: list[str] = []
    for raw in [configured_url, *(candidates_csv or "").split(",")]:
        url = normalize_backend_url(raw)
        if url and url not in urls:
            urls.append(url)
    return urls


async def select_backend_url(
    candidates: list[str],
    ping_path: str = "/ping",
    headers: dict[str, str] | None = None,
    probe: t.Callable[[str, str, dict[str, str] | None], t.Awaitable[tuple[bool, str]]] | None = None,
) -> tuple[str | None, list[dict[str, str]]]:
    if probe is None:
        raise ValueError("probe is required")

    attempts: list[dict[str, str]] = []
    for candidate in candidates:
        ok, detail = await probe(candidate, ping_path, headers)
        attempts.append({"url": candidate, "ok": str(bool(ok)).lower(), "detail": str(detail)})
        if ok:
            return candidate, attempts
    return None, attempts
