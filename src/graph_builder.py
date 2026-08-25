"""
Flow-Similarity Graph Construction Module.

Constructs PyTorch Geometric Data objects where:
- Each node represents a single network flow.
- Node features (x) are the processed numerical flow features.
- Edges connect each node to its k nearest neighbors based on cosine similarity.
- Edge weights represent the cosine similarity between connected nodes.

Uses sklearn NearestNeighbors for efficient k-NN computation (avoids full pairwise matrix).
Designed for memory-constrained environments (6 GB VRAM, 16 GB RAM laptop).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as sklearn_normalize

logger = logging.getLogger(__name__)

# Lazy import to avoid hard failure if torch_geometric is not installed
try:
    from torch_geometric.data import Data
except ImportError:
    Data = None
    logger.warning(
        "torch_geometric not installed. Graph construction will fail at runtime."
    )


class NetworkGraphBuilder:
    """
    Constructs flow-similarity graphs from tabular network flow data.

    Each flow becomes a graph node. Edges are formed between each node and
    its k nearest neighbors in cosine similarity space. The resulting graph
    is returned as a PyTorch Geometric ``Data`` object.

    Parameters
    ----------
    k_neighbors : int
        Number of nearest neighbors per node. Default 5.
    similarity_metric : str
        Distance metric for NearestNeighbors. Default ``"cosine"``.
    include_self_loops : bool
        Whether to include self-loop edges. Default ``False``.
    graph_size : int
        Maximum number of nodes per graph snapshot. Default 1000.
    random_state : int
        Seed for reproducibility. Default 42.
    n_jobs : int
        Parallelism for NearestNeighbors. Default 1.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        similarity_metric: str = "cosine",
        include_self_loops: bool = False,
        graph_size: int = 1000,
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")

        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.include_self_loops = include_self_loops
        self.graph_size = graph_size
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Fitted state
        self._nn_model: Optional[NearestNeighbors] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Input normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(data: Union[np.ndarray, torch.Tensor, "pd.DataFrame"]) -> np.ndarray:
        """Convert input to a 2-D float32 numpy array."""
        if isinstance(data, torch.Tensor):
            arr = data.detach().cpu().numpy()
        elif hasattr(data, "values"):
            # pandas DataFrame or Series
            arr = data.values
        elif isinstance(data, np.ndarray):
            arr = data
        else:
            arr = np.asarray(data)

        arr = arr.astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    @staticmethod
    def _to_labels(
        labels: Optional[Union[np.ndarray, torch.Tensor, list, "pd.Series"]],
    ) -> Optional[np.ndarray]:
        """Convert optional labels to a 1-D int64 numpy array."""
        if labels is None:
            return None
        if isinstance(labels, torch.Tensor):
            return labels.detach().cpu().numpy().astype(np.int64).ravel()
        if hasattr(labels, "values"):
            return labels.values.astype(np.int64).ravel()
        return np.asarray(labels, dtype=np.int64).ravel()

    # ------------------------------------------------------------------
    # Core k-NN graph construction
    # ------------------------------------------------------------------

    def _build_knn_edges(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build k-NN edge index and edge weights from feature matrix.

        Parameters
        ----------
        features : np.ndarray
            (N, D) feature matrix.

        Returns
        -------
        edge_index : np.ndarray of shape (2, E)
            COO-format directed edges.
        edge_weights : np.ndarray of shape (E,)
            Cosine similarity weights (1 - cosine_distance) in [0, 1].
        """
        n_samples = features.shape[0]
        effective_k = min(self.k_neighbors, n_samples - 1)

        if effective_k < 1:
            # Fewer than 2 nodes → no edges possible
            return (
                np.empty((2, 0), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )

        # sklearn NearestNeighbors with cosine metric uses brute-force by
        # default, which is efficient enough for graph_size ≤ ~10 000 and
        # avoids the overhead of tree construction.
        nn = NearestNeighbors(
            n_neighbors=effective_k + 1,  # +1 because the query itself is returned
            metric=self.similarity_metric,
            algorithm="brute",
            n_jobs=self.n_jobs,
        )
        nn.fit(features)
        distances, indices = nn.kneighbors(features)

        # distances from cosine metric are *cosine distances* in [0, 2].
        # Cosine similarity = 1 - cosine_distance.

        src_list = []
        dst_list = []
        weight_list = []

        for node_idx in range(n_samples):
            for j in range(distances.shape[1]):
                neighbor_idx = indices[node_idx, j]
                if neighbor_idx == node_idx and not self.include_self_loops:
                    continue
                cos_dist = distances[node_idx, j]
                similarity = max(1.0 - cos_dist, 0.0)  # clamp to [0, 1]

                src_list.append(node_idx)
                dst_list.append(neighbor_idx)
                weight_list.append(similarity)

        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        edge_weights = np.array(weight_list, dtype=np.float32)

        return edge_index, edge_weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        features: Union[np.ndarray, torch.Tensor, Any],
    ) -> "NetworkGraphBuilder":
        """
        Fit the internal NearestNeighbors model on a reference feature set.

        This is useful when you want to reuse the same k-NN index for
        multiple queries.

        Parameters
        ----------
        features : array-like of shape (N, D)
            Numerical flow features (labels must NOT be included).

        Returns
        -------
        self
        """
        X = self._to_numpy(features)
        effective_k = min(self.k_neighbors, X.shape[0] - 1)
        if effective_k < 1:
            logger.warning("Fewer than 2 samples; k-NN model cannot be fitted meaningfully.")
            self._nn_model = None
            self._is_fitted = True
            return self

        self._nn_model = NearestNeighbors(
            n_neighbors=effective_k + 1,
            metric=self.similarity_metric,
            algorithm="brute",
            n_jobs=self.n_jobs,
        )
        self._nn_model.fit(X)
        self._is_fitted = True
        return self

    def build_graph(
        self,
        features: Union[np.ndarray, torch.Tensor, Any],
        labels: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
    ) -> "Data":
        """
        Construct a PyTorch Geometric Data object from flow features.

        Parameters
        ----------
        features : array-like of shape (N, D)
            Numerical flow features. Labels must NOT be included.
        labels : array-like of shape (N,), optional
            Integer class labels. Stored as ``data.y`` but NOT used for
            similarity computation.

        Returns
        -------
        torch_geometric.data.Data
            Graph with attributes ``x``, ``edge_index``, ``edge_weight``,
            and optionally ``y``.
        """
        if Data is None:
            raise ImportError(
                "torch_geometric is required for graph construction. "
                "Install it with: pip install torch_geometric"
            )

        X = self._to_numpy(features)
        y = self._to_labels(labels)

        if y is not None and len(y) != X.shape[0]:
            raise ValueError(
                f"Length mismatch: features has {X.shape[0]} rows but "
                f"labels has {len(y)} entries."
            )

        edge_index_np, edge_weights_np = self._build_knn_edges(X)

        data = Data(
            x=torch.tensor(X, dtype=torch.float32),
            edge_index=torch.tensor(edge_index_np, dtype=torch.long),
            edge_weight=torch.tensor(edge_weights_np, dtype=torch.float32),
        )

        if y is not None:
            data.y = torch.tensor(y, dtype=torch.long)

        data.num_nodes = X.shape[0]

        return data

    def build_graph_from_flows(
        self,
        flow_features: Union[np.ndarray, torch.Tensor, Any],
        labels: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
    ) -> "Data":
        """
        Alias for ``build_graph`` for backward compatibility.

        Parameters
        ----------
        flow_features : array-like of shape (N, D)
        labels : array-like of shape (N,), optional

        Returns
        -------
        torch_geometric.data.Data
        """
        return self.build_graph(flow_features, labels)

    def build_batch_graphs(
        self,
        features: Union[np.ndarray, torch.Tensor, Any],
        labels: Optional[Union[np.ndarray, torch.Tensor, list]] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
    ) -> List["Data"]:
        """
        Split a large dataset into manageable graph snapshots.

        Each snapshot contains at most ``batch_size`` (default
        ``self.graph_size``) nodes with their own k-NN edges.

        Parameters
        ----------
        features : array-like of shape (N, D)
            Full feature matrix.
        labels : array-like of shape (N,), optional
            Full label vector.
        batch_size : int, optional
            Nodes per graph. Defaults to ``self.graph_size``.
        shuffle : bool
            Whether to shuffle rows before batching. Default True.

        Returns
        -------
        list of Data
            One PyTorch Geometric Data object per batch.
        """
        X = self._to_numpy(features)
        y = self._to_labels(labels)
        bs = batch_size or self.graph_size
        n_samples = X.shape[0]

        # Deterministic shuffle
        rng = np.random.RandomState(self.random_state)
        if shuffle:
            perm = rng.permutation(n_samples)
            X = X[perm]
            if y is not None:
                y = y[perm]

        graphs: List[Data] = []
        for start in range(0, n_samples, bs):
            end = min(start + bs, n_samples)
            X_batch = X[start:end]
            y_batch = y[start:end] if y is not None else None
            graph = self.build_graph(X_batch, y_batch)
            graphs.append(graph)

        logger.info(
            "Built %d batch graphs (batch_size=%d) from %d total flows.",
            len(graphs), bs, n_samples,
        )
        return graphs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_graph(
        self,
        graph: "Data",
        file_path: Union[str, Path],
    ) -> Path:
        """
        Save a PyTorch Geometric Data object to disk.

        Parameters
        ----------
        graph : Data
            The graph to save.
        file_path : str or Path
            Destination path (will use ``.pt`` extension).

        Returns
        -------
        Path
            Resolved path to the saved file.
        """
        path = Path(file_path)
        if path.suffix != ".pt":
            path = path.with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph, str(path))
        logger.info("Saved graph to %s", path)
        return path

    @staticmethod
    def load_graph(file_path: Union[str, Path]) -> "Data":
        """
        Load a PyTorch Geometric Data object from disk.

        Parameters
        ----------
        file_path : str or Path
            Path to a ``.pt`` file saved by ``save_graph``.

        Returns
        -------
        Data
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")
        graph = torch.load(str(path), weights_only=False)
        logger.info("Loaded graph from %s", path)
        return graph

    # ------------------------------------------------------------------
    # Utility / introspection
    # ------------------------------------------------------------------

    @staticmethod
    def graph_summary(graph: "Data") -> Dict[str, Any]:
        """
        Return a human-readable summary dict for a PyTorch Geometric Data object.
        """
        num_nodes = graph.num_nodes
        num_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0
        num_features = graph.x.shape[1] if graph.x is not None else 0
        avg_degree = num_edges / max(num_nodes, 1)

        summary = {
            "num_nodes": num_nodes,
            "num_features": num_features,
            "num_edges": num_edges,
            "avg_degree": round(avg_degree, 2),
        }

        if hasattr(graph, "y") and graph.y is not None:
            unique, counts = torch.unique(graph.y, return_counts=True)
            summary["class_distribution"] = {
                int(u): int(c) for u, c in zip(unique, counts)
            }

        if hasattr(graph, "edge_weight") and graph.edge_weight is not None:
            ew = graph.edge_weight
            summary["edge_weight_stats"] = {
                "min": float(ew.min()),
                "max": float(ew.max()),
                "mean": float(ew.mean()),
                "std": float(ew.std()),
            }

        return summary
