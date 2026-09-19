"""
FAISS Vector Index for Attendify.
Provides fast inner-product (cosine similarity) nearest neighbor search
over 512-dimensional face embeddings.
"""
import faiss
import numpy as np

EMBEDDING_DIM = 512

class FaceVectorIndex:
    def __init__(self, dim=EMBEDDING_DIM):
        self.dim = dim
        self.index = faiss.IndexFlatIP(self.dim)
        # Vector index offset -> student_id mapping
        self.id_map = []

    def clear(self):
        self.index.reset()
        self.id_map = []

    def add(self, embeddings: list, student_id: int):
        """
        Add one or multiple 512-d embeddings for a given student.
        Embeddings are L2-normalized so Inner Product == Cosine Similarity.
        """
        if not embeddings:
            return

        arr = np.array(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        faiss.normalize_L2(arr)
        self.index.add(arr)
        for _ in range(arr.shape[0]):
            self.id_map.append(student_id)

    def search(self, probe_embedding: np.ndarray, top_k=5):
        """
        Search for nearest matches.
        Returns list of (student_id, score) sorted by score desc.
        """
        if self.index.ntotal == 0:
            return []

        probe = np.array(probe_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(probe)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(probe, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.id_map):
                results.append((self.id_map[idx], float(score)))
        return results

    def find_best_match(self, probe_embedding: np.ndarray, threshold: float = 0.68):
        """
        Returns (best_student_id or None, best_score).
        """
        results = self.search(probe_embedding, top_k=5)
        if not results:
            return None, -1.0

        # Group by student_id to find max score for each student
        student_scores = {}
        for student_id, score in results:
            if student_id not in student_scores or score > student_scores[student_id]:
                student_scores[student_id] = score

        best_student_id = max(student_scores, key=student_scores.get)
        best_score = student_scores[best_student_id]

        if best_score < threshold:
            return None, best_score

        return best_student_id, best_score

vector_index = FaceVectorIndex()
