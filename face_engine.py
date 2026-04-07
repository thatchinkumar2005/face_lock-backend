import os
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

EMBEDDINGS_DIR = Path("embeddings")
EMBEDDINGS_DIR.mkdir(exist_ok=True)
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "users.pkl"

SIMILARITY_THRESHOLD = 0.75


class FaceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # MTCNN detects and crops faces
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=40,
            keep_all=False,
            device=self.device,
        )
        # InceptionResnetV1 produces 512-d embeddings
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.user_embeddings: dict[str, np.ndarray] = self._load_embeddings()

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def _load_embeddings(self) -> dict:
        if EMBEDDINGS_FILE.exists():
            with open(EMBEDDINGS_FILE, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_embeddings(self):
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(self.user_embeddings, f)

    # ------------------------------------------------------------------ #
    #  Core helpers                                                        #
    # ------------------------------------------------------------------ #

    def _get_embedding(self, image: Image.Image) -> np.ndarray | None:
        """Detect face, crop it, and return its 512-d embedding."""
        img_tensor = self.mtcnn(image)
        if img_tensor is None:
            return None
        with torch.no_grad():
            embedding = self.resnet(img_tensor.unsqueeze(0).to(self.device))
        return embedding.cpu().numpy()[0]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def train(self, user_id: str, images: list[Image.Image]) -> dict:
        """
        Compute embeddings for all supplied images, average them, and
        store as the canonical embedding for user_id.
        Returns {"success": bool, "photos_used": int, "message": str}
        """
        embeddings = []
        for img in images:
            emb = self._get_embedding(img)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return {
                "success": False,
                "photos_used": 0,
                "message": "No faces detected in any of the supplied images.",
            }

        mean_embedding = np.mean(embeddings, axis=0)
        # L2-normalise so cosine similarity == dot product
        mean_embedding /= np.linalg.norm(mean_embedding)
        self.user_embeddings[user_id] = mean_embedding
        self._save_embeddings()

        return {
            "success": True,
            "photos_used": len(embeddings),
            "message": f"Trained '{user_id}' with {len(embeddings)} photo(s).",
        }

    def verify(self, image: Image.Image) -> dict:
        """
        Compare the face in image against all stored embeddings.
        Returns {"status": "open"|"denied", "user": str|None, "confidence": float}
        """
        if not self.user_embeddings:
            return {
                "status": "denied",
                "user": None,
                "confidence": 0.0,
                "message": "No users trained yet.",
            }

        query_emb = self._get_embedding(image)
        if query_emb is None:
            return {
                "status": "denied",
                "user": None,
                "confidence": 0.0,
                "message": "No face detected in the image.",
            }

        query_emb = query_emb / np.linalg.norm(query_emb)

        best_user = None
        best_score = -1.0
        for uid, stored_emb in self.user_embeddings.items():
            score = self._cosine_similarity(query_emb, stored_emb)
            if score > best_score:
                best_score = score
                best_user = uid

        if best_score >= SIMILARITY_THRESHOLD:
            return {
                "status": "open",
                "user": best_user,
                "confidence": round(best_score, 4),
                "message": f"Access granted for '{best_user}'.",
            }
        return {
            "status": "denied",
            "user": None,
            "confidence": round(best_score, 4),
            "message": "Face not recognised.",
        }

    def list_users(self) -> list[str]:
        return list(self.user_embeddings.keys())

    def delete_user(self, user_id: str) -> bool:
        if user_id in self.user_embeddings:
            del self.user_embeddings[user_id]
            self._save_embeddings()
            return True
        return False
