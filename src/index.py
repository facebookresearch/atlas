# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
import os
import pickle
from typing import Optional, Set, Tuple, Union, Any
import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch
from src import dist_utils
from src.retrievers import EMBEDDINGS_DIM

FAISSGPUIndex = Union[
    faiss.GpuIndexIVFFlat, faiss.GpuIndexIVFPQ, faiss.GpuIndexIVFScalarQuantizer, faiss.GpuIndexFlatIP
]
FAISSIndex = Union[FAISSGPUIndex, faiss.IndexPQ]

GPUIndexConfig = Union[
    faiss.GpuIndexIVFPQConfig,
    faiss.GpuIndexIVFFlatConfig,
    faiss.GpuIndexIVFScalarQuantizerConfig,
    faiss.GpuIndexFlatConfig,
]
BITS_PER_CODE: int = 8
CHUNK_SPLIT: int = 3


def serialize_listdocs(ids):
    ids = pickle.dumps(ids)
    ids = torch.tensor(list(ids), dtype=torch.uint8).cuda()
    return ids


def deserialize_listdocs(ids):
    return [pickle.loads(x.cpu().numpy().tobytes()) for x in ids]


class DistributedIndex(object):
    def __init__(self):
        self.embeddings = None
        self.doc_map = dict()
        self.is_in_gpu = True

    def init_embeddings(self, passages, dim: Optional[int] = EMBEDDINGS_DIM):
        self.doc_map = {i: doc for i, doc in enumerate(passages)}
        self.embeddings = torch.zeros(dim, (len(passages)), dtype=torch.float16)
        if self.is_in_gpu:
            self.embeddings = self.embeddings.cuda()

    def _get_saved_embedding_path(self, save_dir: str, shard: int) -> str:
        return os.path.join(save_dir, f"embeddings.{shard}.pt")

    def _get_saved_passages_path(self, save_dir: str, shard: int) -> str:
        return os.path.join(save_dir, f"passages.{shard}.pt")

    def save_index(self, path: str, total_saved_shards: int, overwrite_saved_passages: bool = False) -> None:
        """
        Saves index state to disk, which can later be loaded by the load_index method.
        Specifically, it saves the embeddings and passages into total_saved_shards separate file shards.
        This option enables loading the index in another session with a different number of workers, as long as the number of workers is divisible by total_saved_shards.
        Note that the embeddings will always be saved to disk (it will overwrite any embeddings previously saved there).
        The passages will only be saved to disk if they have not already been written to the save directory before, unless the option --overwrite_saved_passages is passed.
        """
        assert self.embeddings is not None
        rank = dist_utils.get_rank()
        ws = dist_utils.get_world_size()
        assert total_saved_shards % ws == 0, f"N workers must be a multiple of shards to save"
        shards_per_worker = total_saved_shards // ws
        n_embeddings = self.embeddings.shape[1]
        embeddings_per_shard = math.ceil(n_embeddings / shards_per_worker)
        assert n_embeddings == len(self.doc_map), len(self.doc_map)
        for shard_ind, (shard_start) in enumerate(range(0, n_embeddings, embeddings_per_shard)):
            shard_end = min(shard_start + embeddings_per_shard, n_embeddings)
            shard_id = shard_ind + rank * shards_per_worker  # get global shard number
            passage_shard_path = self._get_saved_passages_path(path, shard_id)
            if not os.path.exists(passage_shard_path) or overwrite_saved_passages:
                passage_shard = [self.doc_map[i] for i in range(shard_start, shard_end)]
                with open(passage_shard_path, "wb") as fobj:
                    pickle.dump(passage_shard, fobj, protocol=pickle.HIGHEST_PROTOCOL)
            embeddings_shard = self.embeddings[:, shard_start:shard_end]
            embedding_shard_path = self._get_saved_embedding_path(path, shard_id)
            torch.save(embeddings_shard, embedding_shard_path)

    def load_index(self, path: str, total_saved_shards: int):
        """
        Loads sharded embeddings and passages files (no index is loaded).
        """
        rank = dist_utils.get_rank()
        ws = dist_utils.get_world_size()
        assert total_saved_shards % ws == 0, f"N workers must be a multiple of shards to save"
        shards_per_worker = total_saved_shards // ws
        passages = []
        embeddings = []
        for shard_id in range(rank * shards_per_worker, (rank + 1) * shards_per_worker):
            passage_shard_path = self._get_saved_passages_path(path, shard_id)
            with open(passage_shard_path, "rb") as fobj:
                passages.append(pickle.load(fobj))
            embeddings_shard_path = self._get_saved_embedding_path(path, shard_id)
            embeddings.append(torch.load(embeddings_shard_path, map_location="cpu").cuda())
        self.doc_map = {}
        n_passages = 0
        for chunk in passages:
            for p in chunk:
                self.doc_map[n_passages] = p
                n_passages += 1
        self.embeddings = torch.concat(embeddings, dim=1)

    def _compute_scores_and_indices(self, allqueries: torch.tensor, topk: int) -> Tuple[torch.tensor, torch.tensor]:
        """
        Computes the distance matrix for the query embeddings and embeddings chunk and returns the k-nearest neighbours and corresponding scores.
        """
        scores = torch.matmul(allqueries.half(), self.embeddings)
        scores, indices = torch.topk(scores, topk, dim=1)

        return scores, indices

    @torch.no_grad()
    def search_knn(self, queries, topk):
        """
        Conducts exhaustive search of the k-nearest neighbours using the inner product metric.
        """
        allqueries = dist_utils.varsize_all_gather(queries)
        allsizes = dist_utils.get_varsize(queries)
        allsizes = np.cumsum([0] + allsizes.cpu().tolist())
        # compute scores for the part of the index located on each process
        scores, indices = self._compute_scores_and_indices(allqueries, topk)
        indices = indices.tolist()
        docs = [[self.doc_map[x] for x in sample_indices] for sample_indices in indices]
        if torch.distributed.is_initialized():
            docs = [docs[allsizes[k] : allsizes[k + 1]] for k in range(len(allsizes) - 1)]
            docs = [serialize_listdocs(x) for x in docs]
            scores = [scores[allsizes[k] : allsizes[k + 1]] for k in range(len(allsizes) - 1)]
            gather_docs = [dist_utils.varsize_gather(docs[k], dst=k, dim=0) for k in range(dist_utils.get_world_size())]
            gather_scores = [
                dist_utils.varsize_gather(scores[k], dst=k, dim=1) for k in range(dist_utils.get_world_size())
            ]
            rank_scores = gather_scores[dist_utils.get_rank()]
            rank_docs = gather_docs[dist_utils.get_rank()]
            scores = torch.cat(rank_scores, dim=1)
            rank_docs = deserialize_listdocs(rank_docs)
            merge_docs = [[] for _ in range(queries.size(0))]
            for docs in rank_docs:
                for k, x in enumerate(docs):
                    merge_docs[k].extend(x)
            docs = merge_docs
        _, subindices = torch.topk(scores, topk, dim=1)
        scores = scores.tolist()
        subindices = subindices.tolist()
        # Extract topk scores and associated ids
        scores = [[scores[k][j] for j in idx] for k, idx in enumerate(subindices)]
        docs = [[docs[k][j] for j in idx] for k, idx in enumerate(subindices)]
        return docs, scores

    def is_index_trained(self) -> bool:
        return True


class DistributedFAISSIndex(DistributedIndex):
    def __init__(self, index_type: str, code_size: Optional[int] = None):
        super().__init__()
        self.embeddings = None
        self.doc_map = dict()
        self.faiss_gpu_index = None
        self.gpu_resources = None
        self.faiss_index_trained = False
        self.faiss_index_type = index_type
        self.code_size = code_size
        self.is_in_gpu = False

    def _get_faiss_index_filename(self, save_index_path: str) -> str:
        """
        Creates the filename to save the trained index to using the index type, code size (if not None) and rank.
        """
        rank = dist_utils.get_rank()
        if self.code_size:
            return save_index_path + f"/index{self.faiss_index_type}_{str(self.code_size)}_rank_{rank}.faiss"
        return save_index_path + f"/index{self.faiss_index_type}_rank_{rank}.faiss"

    def _add_embeddings_to_gpu_index(self) -> None:
        """
        Add embeddings to index and sets the nprobe parameter.
        """
        assert self.faiss_gpu_index is not None, "The FAISS GPU index was not correctly instantiated."
        assert self.faiss_gpu_index.is_trained == True, "The FAISS index has not been trained."
        if self.faiss_gpu_index.ntotal == 0:
            self._add_embeddings_by_chunks()

    def _add_embeddings_by_chunks(self) -> None:
        _, num_points = self.embeddings.shape
        chunk_size = num_points // CHUNK_SPLIT
        split_embeddings = [
            self.embeddings[:, 0:chunk_size],
            self.embeddings[:, chunk_size : 2 * chunk_size],
            self.embeddings[:, 2 * chunk_size : num_points],
        ]
        for embeddings_chunk in split_embeddings:
            if isinstance(self.faiss_gpu_index, FAISSGPUIndex.__args__):
                self.faiss_gpu_index.add(self._cast_to_torch32(embeddings_chunk.T))
            else:
                self.faiss_gpu_index.add(self._cast_to_numpy(embeddings_chunk.T))

    def _compute_scores_and_indices(self, allqueries: torch.tensor, topk: int) -> Tuple[torch.tensor, torch.tensor]:
        """
        Computes the distance matrix for the query embeddings and embeddings chunk and returns the k-nearest neighbours and corresponding scores.
        """
        _, num_points = self.embeddings.shape
        self.faiss_gpu_index.nprobe = math.floor(math.sqrt(num_points))
        self._add_embeddings_to_gpu_index()
        if isinstance(self.faiss_gpu_index, FAISSGPUIndex.__args__):
            scores, indices = self.faiss_gpu_index.search(self._cast_to_torch32(allqueries), topk)
        else:
            np_scores, indices = self.faiss_gpu_index.search(self._cast_to_numpy(allqueries), topk)
            scores = torch.from_numpy(np_scores).cuda()
        return scores.half(), indices

    def save_index(self, save_index_path: str, save_index_n_shards: int) -> None:
        """
        Saves the embeddings and passages and if there is a FAISS index, it saves it.
        """
        super().save_index(save_index_path, save_index_n_shards)
        self._save_faiss_index(save_index_path)

    def _save_faiss_index(self, path: str) -> None:
        """
        Moves the GPU FAISS index to CPU and saves it to a .faiss file.
        """
        index_path = self._get_faiss_index_filename(path)
        assert self.faiss_gpu_index is not None, "There is no FAISS index to save."
        cpu_index = faiss.index_gpu_to_cpu(self.faiss_gpu_index)
        faiss.write_index(cpu_index, index_path)

    def _load_faiss_index(self, load_index_path: str) -> None:
        """
        Loads a FAISS index and moves it to the GPU.
        """
        faiss_cpu_index = faiss.read_index(load_index_path)
        # move to GPU
        self._move_index_to_gpu(faiss_cpu_index)

    def load_index(self, path: str, total_saved_shards: int) -> None:
        """
        Loads passage embeddings and passages and a faiss index (if it exists).
        Otherwise, it initialises and trains the index in the GPU with GPU FAISS.
        """
        super().load_index(path, total_saved_shards)
        load_index_path = self._get_faiss_index_filename(path)
        if os.path.exists(load_index_path):
            self._load_faiss_index(load_index_path)
        else:
            self.train_index()

    def is_index_trained(self) -> bool:
        if self.faiss_gpu_index is None:
            return self.faiss_index_trained
        return not self.faiss_gpu_index.is_trained

    def _initialise_index(self) -> None:
        """
        Initialises the index in the GPU with GPU FAISS.
        Supported gpu index types: IVFFlat, IndexFlatIP, IndexIVFPQ, IVFSQ.
        """
        dimension, num_points = self.embeddings.shape
        # @TODO: Add support to set the n_list and n_probe parameters.
        n_list = math.floor(math.sqrt(num_points))
        self.faiss_gpu_index = self.gpu_index_factory(dimension, n_list)

    @torch.no_grad()
    def _set_gpu_options(self) -> faiss.GpuMultipleClonerOptions:
        """
        Returns the GPU cloner options neccessary when moving a CPU index to the GPU.
        """
        cloner_opts = faiss.GpuClonerOptions()
        cloner_opts.useFloat16 = True
        cloner_opts.usePrecomputed = False
        cloner_opts.indicesOptions = faiss.INDICES_32_BIT
        return cloner_opts

    @torch.no_grad()
    def _set_index_config_options(self, index_config: GPUIndexConfig) -> GPUIndexConfig:
        """
        Returns the GPU config options for GPU indexes.
        """
        index_config.device = torch.cuda.current_device()
        index_config.indicesOptions = faiss.INDICES_32_BIT
        index_config.useFloat16 = True

        return index_config

    def _create_PQ_index(self, dimension) -> FAISSIndex:
        """
        GPU config options for PQ index
        """
        cpu_index = faiss.index_factory(dimension, "PQ" + str(self.code_size), faiss.METRIC_INNER_PRODUCT)
        cfg = self._set_gpu_options()
        return faiss.index_cpu_to_gpu(self.gpu_resources, self.embeddings.get_device(), cpu_index, cfg)

    @torch.no_grad()
    def gpu_index_factory(self, dimension: int, n_list: Optional[int] = None) -> FAISSIndex:
        """
        Instantiates and returns the selected GPU index class.
        """
        self.gpu_resources = faiss.StandardGpuResources()
        if self.faiss_index_type == "ivfflat":
            config = self._set_index_config_options(faiss.GpuIndexIVFFlatConfig())
            return faiss.GpuIndexIVFFlat(
                self.gpu_resources,
                dimension,
                n_list,
                faiss.METRIC_INNER_PRODUCT,
                config,
            )
        elif self.faiss_index_type == "flat":
            config = self._set_index_config_options(faiss.GpuIndexFlatConfig())
            return faiss.GpuIndexFlatIP(self.gpu_resources, dimension, config)
        elif self.faiss_index_type == "pq":
            return self._create_PQ_index(dimension)
        elif self.faiss_index_type == "ivfpq":
            config = self._set_index_config_options(faiss.GpuIndexIVFPQConfig())
            return faiss.GpuIndexIVFPQ(
                self.gpu_resources,
                dimension,
                n_list,
                self.code_size,
                BITS_PER_CODE,
                faiss.METRIC_INNER_PRODUCT,
                config,
            )
        elif self.faiss_index_type == "ivfsq":
            config = self._set_index_config_options(faiss.GpuIndexIVFScalarQuantizerConfig())
            qtype = faiss.ScalarQuantizer.QT_4bit
            return faiss.GpuIndexIVFScalarQuantizer(
                self.gpu_resources,
                dimension,
                n_list,
                qtype,
                faiss.METRIC_INNER_PRODUCT,
                True,
                config,
            )
        else:
            raise ValueError("unsupported index type")

    @torch.no_grad()
    def train_index(self) -> None:
        """
        It initialises the index and trains it according to the refresh index schedule.
        """
        if self.faiss_gpu_index is None:
            self._initialise_index()
        self.faiss_gpu_index.reset()
        if isinstance(self.faiss_gpu_index, FAISSGPUIndex.__args__):
            self.faiss_gpu_index.train(self._cast_to_torch32(self.embeddings.T))
        else:
            self.faiss_gpu_index.train(self._cast_to_numpy(self.embeddings.T))

    @torch.no_grad()
    def _cast_to_torch32(self, embeddings: torch.tensor) -> torch.tensor:
        """
        Converts a torch tensor to a contiguous float 32 torch tensor.
        """
        return embeddings.type(torch.float32).contiguous()

    @torch.no_grad()
    def _cast_to_numpy(self, embeddings: torch.tensor) -> np.ndarray:
        """
        Converts a torch tensor to a contiguous numpy float 32 ndarray.
        """
        return embeddings.cpu().to(dtype=torch.float16).numpy().astype("float32").copy(order="C")

    @torch.no_grad()
    def _move_index_to_gpu(self, cpu_index: FAISSIndex) -> None:
        """
        Moves a loaded index to GPU.
        """
        self.gpu_resources = faiss.StandardGpuResources()
        cfg = self._set_gpu_options()
        self.faiss_gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources, torch.cuda.current_device(), cpu_index, cfg)
