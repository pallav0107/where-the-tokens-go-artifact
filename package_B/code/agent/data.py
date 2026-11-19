"""HotpotQA distractor validation loading, sampling and KB construction (PREREG section 4)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pyarrow.parquet as pq

SAMPLE_SEED = 20261115
_DOC_ID_LEN = 10
_CHUNK_SENTENCES = 3


@dataclass
class Question:
    id: str
    question: str
    answer: str
    position: int  # index in the sampled order (0-based)
    split: str  # "test" iff position % 3 == 2, else "train"
    block: int | None = None  # PREREG_v3: index of the question's 48-question KB block


@dataclass
class KB:
    docs: list  # [{doc_id, title, paragraph}]
    chunks: list  # [{doc_id, title, text}] where text = "Title: s1 s2 s3"


@dataclass
class Dataset:
    sample: list  # sampled questions in sampled order
    smoke: list  # smoke-test questions (disjoint from sample)
    kb: KB
    dataset_sha256: str
    n_rows: int
    dev: list = field(default_factory=list)  # PREREG_v2 dev split (positions after smoke)
    dev_kb: KB | None = None  # the dev split's own knowledge base
    blocks: list = field(default_factory=list)  # PREREG_v3: [(questions, KB)] of one split


def _norm_ws(text: str) -> str:
    return " ".join(str(text).split())


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, path: str) -> None:
    """HTTP download with resume (Range header)."""
    have = os.path.getsize(path) if os.path.exists(path) else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            if have and resp.status == 200:
                have = 0  # server ignored the range: restart from scratch
            mode = "ab" if have else "wb"
            with open(path, mode) as fh:
                shutil.copyfileobj(resp, fh)
    except urllib.error.HTTPError as exc:
        if have and exc.code == 416:  # range beyond EOF: file already complete
            return
        raise


def _ensure_parquet(url: str, sha256: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    name = url.rstrip("/").split("/")[-1] or "data.parquet"
    path = os.path.join(cache_dir, name)
    last_err = None
    for attempt in range(5):
        try:
            _download(url, path)
        except Exception as exc:  # noqa: BLE001 - retried, then raised
            last_err = exc
            time.sleep(min(2 ** attempt, 30))
            continue
        if _sha256_file(path) == (sha256 or "").lower():
            return path
        os.remove(path)  # corrupt or mismatched: start over
        last_err = RuntimeError(f"sha256 mismatch for {path}")
    raise RuntimeError(f"could not download {url}: {last_err}")


def _load_rows(cfg: dict) -> tuple[list, str]:
    if cfg.get("parquet_url"):
        path = _ensure_parquet(cfg["parquet_url"], cfg.get("sha256") or "",
                               os.path.expanduser(cfg["cache_dir"]))
        return pq.read_table(path).to_pylist(), (cfg.get("sha256") or "").lower()
    path = cfg["fixture"]
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()], _sha256_file(path)


def _question(row: dict, position: int, split: str | None = None) -> Question:
    return Question(
        id=str(row["id"]),
        question=str(row["question"]),
        answer=str(row["answer"]),
        position=position,
        split=split or ("test" if position % 3 == 2 else "train"),
    )


def build_kb(sample_rows: list) -> KB:
    """Union of context paragraphs of the sampled questions, dedup by title (first wins)."""
    docs: list[dict] = []
    seen: set[str] = set()
    for row in sample_rows:
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        for raw_title, raw_sents in zip(titles, sentences):
            title = _norm_ws(raw_title)
            if not title or title in seen:
                continue
            seen.add(title)
            sents = [_norm_ws(s) for s in raw_sents if _norm_ws(s)]
            paragraph = _norm_ws(" ".join(sents))
            docs.append({
                "doc_id": hashlib.sha1(title.encode("utf-8")).hexdigest()[:_DOC_ID_LEN],
                "title": title,
                "paragraph": paragraph,
                "_sents": sents,
            })
    chunks: list[dict] = []
    for doc in docs:
        sents = doc.pop("_sents")
        for i in range(0, len(sents), _CHUNK_SENTENCES):
            group = " ".join(sents[i:i + _CHUNK_SENTENCES])
            chunks.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "text": f"{doc['title']}: {group}",
            })
    return KB(docs=docs, chunks=chunks)


def block_positions(spec) -> list:
    """[[start, end], ...] (inclusive permutation positions) -> one position list per block."""
    out = []
    for start, end in spec:
        start, end = int(start), int(end)
        if end < start:
            raise ValueError(f"empty block [{start}, {end}]")
        out.append(list(range(start, end + 1)))
    return out


def split_positions(cfg: dict, split: str) -> list:
    """Every permutation position of a PREREG_v3 split (cfg["blocks"][split])."""
    return [p for blk in block_positions(cfg["blocks"][split]) for p in blk]


def load_dataset(cfg: dict, blocks_split: str | None = None) -> Dataset:
    """Load rows, sort by id, draw the sample/smoke split and build the KB. With blocks_split
    (PREREG_v3), also build that split's blocks: each block's questions at their permutation
    positions and the block's own KB pooled from those questions only (v1 section 4 rule).
    Only the requested split is materialised."""
    rows, sha256 = _load_rows(cfg)
    rows.sort(key=lambda r: str(r["id"]))
    n = len(rows)
    seed = int(cfg.get("sample_seed", SAMPLE_SEED))
    perm = np.random.default_rng(seed).permutation(n)
    n_q = int(cfg["n_questions"])
    n_sm = int(cfg.get("n_smoke", 0))
    sample_idx = [int(i) for i in perm[:n_q]]
    smoke_idx = [int(i) for i in perm[n_q:n_q + n_sm]]
    sample = [_question(rows[i], p) for p, i in enumerate(sample_idx)]
    smoke = [_question(rows[i], n_q + p) for p, i in enumerate(smoke_idx)]
    kb = build_kb([rows[i] for i in sample_idx])
    # PREREG_v2 section 3: dev = the next n_dev permutation positions after the smoke
    # questions (54-77 for 48 + 6 + 24); v1 sampling above is untouched.
    start = n_q + n_sm
    dev_idx = [int(i) for i in perm[start:start + int(cfg.get("n_dev", 0))]]
    dev = [_question(rows[i], start + p, "dev") for p, i in enumerate(dev_idx)]
    dev_kb = build_kb([rows[i] for i in dev_idx]) if dev_idx else None
    blocks = []
    if blocks_split is not None:
        spec = (cfg.get("blocks") or {}).get(blocks_split)
        if not spec:
            raise ValueError(f"dataset.blocks has no split {blocks_split!r}")
        for bi, positions in enumerate(block_positions(spec)):
            if positions[-1] >= n:
                raise ValueError(f"block position {positions[-1]} is beyond {n} rows")
            idx = [int(perm[p]) for p in positions]
            qs = [Question(id=str(rows[i]["id"]), question=str(rows[i]["question"]),
                           answer=str(rows[i]["answer"]), position=p, split=blocks_split,
                           block=bi) for p, i in zip(positions, idx)]
            blocks.append((qs, build_kb([rows[i] for i in idx])))
    return Dataset(sample=sample, smoke=smoke, kb=kb, dataset_sha256=sha256, n_rows=n,
                   dev=dev, dev_kb=dev_kb, blocks=blocks)
