from __future__ import annotations

from sentence_transformers import SentenceTransformer, util


class Reranker:
    """Cross-encoder proxy using sentence-transformers cosine similarity."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model)

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return relative relevancy scores for query-context pairs."""
        if not pairs:
            return []

        queries, contexts = zip(*pairs)
        q_emb = self.model.encode(list(queries), convert_to_tensor=True)
        c_emb = self.model.encode(list(contexts), convert_to_tensor=True)
        scores = util.pytorch_cos_sim(q_emb, c_emb).diag().tolist()
        return scores
