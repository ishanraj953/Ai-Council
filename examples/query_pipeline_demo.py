"""
examples/query_pipeline_demo.py
================================
End-to-end demonstration of the Cost-Optimized Query Processing Pipeline.

Run:
    python examples/query_pipeline_demo.py
"""

import sys
import os
import time
import asyncio

# Ensure repo root is on sys.path when run directly
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ai_council.query_pipeline import (
    QueryPipeline,
    EmbeddingEngine,
    VectorStore,
    TopicClassifier,
    SmartQueryDecomposer,
    ModelRouter,
    TokenOptimizer,
    QueryCache,
)

# ─── ANSI colours ────────────────────────────────────────────────────────────
BOLD  = "\033[1m"
CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED   = "\033[91m"
RESET = "\033[0m"

def hdr(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")

def ok(msg: str)  -> None: print(f"  {GREEN}✓{RESET} {msg}")
def info(msg: str) -> None: print(f"  {YELLOW}→{RESET} {msg}")


# ─── Demo queries ─────────────────────────────────────────────────────────────
DEMO_QUERIES = [
    # (label, query)
    ("Simple factual",
     "What is the capital of France?"),

    ("Code + explain",
     "Explain quicksort, compare it with mergesort, and give Python code"),

    ("Data analysis pipeline",
     "Analyze this stock dataset, predict trends, and explain results"),

    ("Multi-step math",
     "Solve the integral of x^2, verify the result, and explain each step"),

    ("Creative + research",
     "Write a short poem about machine learning and research its history"),
]


# ─── Component demos ──────────────────────────────────────────────────────────

def demo_embedding() -> None:
    hdr("Stage 2 — Embedding Engine")
    engine = EmbeddingEngine.default()
    texts = [
        "write a Python function",
        "what is the capital of France",
        "analyze the stock market data",
    ]
    for text in texts:
        vec = engine.embed(text)
        import numpy as np
        norm = float(np.linalg.norm(vec))
        info(f"'{text[:40]}...' → dim={vec.shape[0]}, norm={norm:.4f}")
        assert abs(norm - 1.0) < 1e-4, "Vector should be unit-norm"
        ok("Unit-norm verified")
    stats = engine.cache_stats()
    info(f"Cache: hits={stats['hits']} misses={stats['misses']} rate={stats['hit_rate']:.0%}")


def demo_classification() -> None:
    hdr("Stages 3-4 — Vector Store + Topic Classifier")
    engine = EmbeddingEngine.default()
    store  = VectorStore(engine)
    store.seed_default_topics()
    clf    = TopicClassifier(engine, store)

    test_cases = [
        ("write a Python quicksort function",               "coding"),
        ("calculate the eigenvalues of a matrix",           "math"),
        ("who invented the telephone",                      "general_qa"),
        ("analyze this dataset and find trends",            "data_analysis"),
        ("why does my code throw AttributeError",           "debugging"),
        ("write a haiku about autumn",                      "creative"),
        ("compare the pros and cons of this approach",      "reasoning"),
        ("gather research papers on transformer models",    "research"),
    ]

    correct = 0
    for query, expected in test_cases:
        result = clf.classify(query)
        match = result.topic == expected
        correct += int(match)
        status = ok if match else lambda m: print(f"  {RED}✗{RESET} {m}")
        status(
            f"'{query[:45]}' → {result.topic} "
            f"(expected={expected}, conf={result.confidence:.2f}, {result.latency_ms:.1f}ms)"
        )

    accuracy = correct / len(test_cases)
    info(f"Classification accuracy: {correct}/{len(test_cases)} = {accuracy:.0%}")
    vs_stats = store.stats()
    info(f"Vector store: {vs_stats['n_vectors']} vectors, {vs_stats['n_topics']} topics, backend={vs_stats['backend']}")


def demo_decomposition() -> None:
    hdr("Stage 5 — Smart Query Decomposer")
    decomposer = SmartQueryDecomposer()

    queries = [
        "What is quicksort?",
        "Explain quicksort, compare it with mergesort, and give Python code",
        "Analyze this stock dataset, predict trends, and explain results",
        "Write a poem, research its topic, and then summarize your findings",
    ]

    for q in queries:
        result = decomposer.decompose(q)
        info(f"Query: '{q[:55]}'")
        info(f"  sub-queries={len(result.sub_queries)} simple={result.is_simple} total_complexity={result.total_complexity}")
        for sq in result.sub_queries:
            ok(f"  [{sq.index}] score={sq.complexity_score} level={sq.complexity_level.value} | '{sq.text[:60]}'")
        info(f"  exec_order={result.execution_order}")
        print()


def demo_routing() -> None:
    hdr("Stage 6 — Model Router (Complexity → Tier)")
    from ai_council.query_pipeline.query_decomposer import SubQuery, ComplexityLevel
    from ai_council.query_pipeline.model_router import ModelTier

    router = ModelRouter.default()

    test_cases = [
        ("What is quicksort?",                  2, "general_qa",   ModelTier.CHEAP),
        ("Implement a binary search in Python", 4, "coding",       ModelTier.MID),
        ("Analyze and compare sorting algorithms with proofs", 8, "reasoning", ModelTier.EXPENSIVE),
        ("List EU capitals",                    1, "general_qa",   ModelTier.CHEAP),
        ("Predict stock market trends",         7, "data_analysis", ModelTier.EXPENSIVE),
    ]

    for text, score, topic, expected_tier in test_cases:
        sq = SubQuery(index=0, text=text, complexity_score=score, topic_hint=topic)
        decision = router.route(sq)
        match = decision.tier == expected_tier
        status = ok if match else lambda m: print(f"  {RED}✗{RESET} {m}")
        status(
            f"score={score} topic={topic} → {decision.tier.value.upper()} ({decision.model_id}) "
            f"conf={decision.confidence:.2f} cost=${decision.cost_estimate_usd:.6f}"
        )

    # Cost savings demo
    from ai_council.query_pipeline.query_decomposer import SmartQueryDecomposer
    dec = SmartQueryDecomposer()
    decomp = dec.decompose("Explain quicksort, compare it with mergesort, and give Python code", topic_hint="coding")
    rr = router.route_all(decomp.sub_queries)
    info(f"Route-all: baseline=${rr.baseline_cost_usd:.6f} optimized=${rr.total_estimated_cost_usd:.6f} savings={rr.savings_pct:.1f}%")
    info(f"  cheap={rr.cheap_count} mid={rr.mid_count} expensive={rr.expensive_count}")


def demo_token_optimizer() -> None:
    hdr("Stage 7 — Token Optimizer")
    opt = TokenOptimizer()

    prompt = (
        "As an AI language model, I'd be happy to explain recursion. "
        "Certainly, recursion is a programming technique where a function calls itself."
    )
    context_chunks = [
        "Recursion is widely used in tree traversal algorithms.",
        "The capital of France is Paris. Paris is a major city in Europe.",
        "Python supports recursion with a default stack depth of 1000.",
        "Fibonacci sequence can be computed recursively.",
        "Machine learning models can be trained on GPUs.",
    ]

    result = opt.optimize(
        query="explain recursion in Python",
        prompt=prompt,
        context_chunks=context_chunks,
        budget_tokens=128,
    )
    info(f"Original tokens: {result.original_tokens}")
    info(f"Optimized tokens: {result.optimized_tokens}")
    ok(f"Compression ratio: {result.compression_ratio:.2%} ({result.tokens_saved} tokens saved)")
    info(f"Chunks: kept={result.chunks_kept} dropped={result.chunks_dropped}")
    info(f"Strategies: {result.strategies_applied}")


def demo_cache() -> None:
    hdr("Bonus — Query Cache")
    cache = QueryCache(max_memory_entries=10, ttl_seconds=60)

    query = "What is the capital of France?"
    assert cache.lookup(query) is None
    ok("Cache miss on first lookup (expected)")

    cache.store(query, {"answer": "Paris"})
    result = cache.lookup(query)
    assert result == {"answer": "Paris"}
    ok("Cache hit after store")

    # Case-insensitive normalisation
    result2 = cache.lookup("  what IS the capital of France?  ")
    assert result2 == {"answer": "Paris"}
    ok("Cache hit with normalised whitespace/case variant")

    stats = cache.stats()
    info(f"Stats: hits={stats.hits} misses={stats.misses} rate={stats.hit_rate:.0%}")


def demo_full_pipeline() -> None:
    hdr("Full Pipeline — End-to-End (stub executor)")
    pipeline = QueryPipeline.build()

    for label, query in DEMO_QUERIES:
        t0 = time.perf_counter()
        result = pipeline.process(query)
        elapsed = (time.perf_counter() - t0) * 1_000

        print(f"\n  {BOLD}{label}{RESET}")
        info(f"Query: '{query[:65]}'")
        if result.classification:
            info(f"Topic: {result.classification.topic} (conf={result.classification.confidence:.2f})")
        if result.decomposition:
            info(f"Sub-queries: {len(result.decomposition.sub_queries)}")
        info(result.cost_report.pretty().replace("\n", "\n  "))
        info(f"Total latency: {elapsed:.1f}ms")
        ok(f"success={result.success} from_cache={result.from_cache}")

    # Second run → cache hits
    print(f"\n  {BOLD}Second run (cache hits){RESET}")
    for label, query in DEMO_QUERIES[:2]:
        result = pipeline.process(query)
        ok(f"'{query[:40]}' from_cache={result.from_cache}")

    info("Pipeline stats:")
    stats = pipeline.get_stats()
    for key, val in stats.items():
        info(f"  {key}: {val}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"{BOLD}{CYAN}AI Council — Cost-Optimized Query Pipeline Demo{RESET}\n")

    demo_embedding()
    demo_classification()
    demo_decomposition()
    demo_routing()
    demo_token_optimizer()
    demo_cache()
    demo_full_pipeline()

    print(f"\n{BOLD}{GREEN}All demo stages completed successfully!{RESET}\n")
