from dataclasses import dataclass
from functools import partial

import torch.nn as nn

from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    ComplexRoPE,
    compute_ffn_hidden_dim,
    Embedding,
    Linear,
    RMSNorm,
)
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
    make_gqa_config,
)
from torchtitan.models.common.param_init import depth_scaled_std, skip_param_init
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock
from torchtitan.models.llama3.parallelize import parallelize_llama
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from .gloeckle import GloeckleModel

HEAD_DIM = 64
FFN_MULTIPLE_OF = 256
ROPE_THETA = 10000.0


@dataclass(frozen=True)
class Shape:
    dim: int
    n_layers: int

    @property
    def n_heads(self) -> int:
        return self.dim // HEAD_DIM

    @property
    def hidden_dim(self) -> int:
        return compute_ffn_hidden_dim(self.dim, multiple_of=FFN_MULTIPLE_OF)

    @property
    def parameters_per_block(self) -> int:
        attention = 4 * self.dim * self.dim
        feed_forward = 3 * self.dim * self.hidden_dim
        norms = 2 * self.dim
        return attention + feed_forward + norms

    @property
    def non_embedding_parameters(self) -> int:
        return self.parameters_per_block * self.n_layers + self.dim


SHAPES: dict[str, Shape] = {
    "debug": Shape(dim=384, n_layers=4),
    "17m": Shape(dim=512, n_layers=5),
    "57m": Shape(dim=768, n_layers=8),
    "101m": Shape(dim=896, n_layers=10),
}

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_TIED_EMBEDDING_INIT = {"weight": skip_param_init}


def _output_linear_init(dim: int) -> dict:
    std = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _block_configs(
    shape: Shape, rope: ComplexRoPE.Config, attn_backend: str
) -> list[Llama3TransformerBlock.Config]:
    inner_attention = get_attention_config(attn_backend)
    return [
        Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(
                normalized_shape=shape.dim, param_init=_NORM_INIT
            ),
            ffn_norm=RMSNorm.Config(normalized_shape=shape.dim, param_init=_NORM_INIT),
            attention=make_gqa_config(
                dim=shape.dim,
                n_heads=shape.n_heads,
                wqkv_param_init=_LINEAR_INIT,
                wo_param_init=_depth_init(layer_id),
                inner_attention=inner_attention,
                rope=rope,
                fuse_qkv=True,
            ),
            feed_forward=make_ffn_config(
                dim=shape.dim,
                hidden_dim=shape.hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            ),
        )
        for layer_id in range(shape.n_layers)
    ]


def model_config(
    shape_name: str,
    *,
    vocab_size: int,
    seq_len: int,
    attn_backend: str = "flex",
) -> Llama3Model.Config:
    shape = SHAPES[shape_name]
    rope = ComplexRoPE.Config(
        dim=HEAD_DIM,
        max_context_length=seq_len,
        theta=ROPE_THETA,
        scaling="none",
    )
    return Llama3Model.Config(
        dim=shape.dim,
        vocab_size=vocab_size,
        enable_weight_tying=True,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=shape.dim,
            param_init=_TIED_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=shape.dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=shape.dim,
            out_features=vocab_size,
            param_init=_output_linear_init(shape.dim),
        ),
        layers=_block_configs(shape, rope, attn_backend),
    )


def model_spec(
    shape_name: str,
    *,
    vocab_size: int,
    seq_len: int,
    attn_backend: str = "flex",
) -> ModelSpec:
    return ModelSpec(
        name="mtp_titan_llama3",
        flavor=shape_name,
        model=model_config(
            shape_name,
            vocab_size=vocab_size,
            seq_len=seq_len,
            attn_backend=attn_backend,
        ),
        max_context_length=seq_len,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )


def gloeckle_model_config(
    shape_name: str,
    *,
    num_heads: int,
    vocab_size: int,
    seq_len: int,
    attn_backend: str = "flex",
) -> GloeckleModel.Config:
    dense = model_config(
        shape_name,
        vocab_size=vocab_size,
        seq_len=seq_len,
        attn_backend=attn_backend,
    )
    trunk_depth = len(dense.layers) - num_heads
    if trunk_depth < 1:
        raise ValueError(
            f"shape {shape_name!r} has {len(dense.layers)} blocks, "
            f"too few for {num_heads} heads"
        )
    return GloeckleModel.Config(
        dim=dense.dim,
        vocab_size=dense.vocab_size,
        enable_weight_tying=dense.enable_weight_tying,
        tok_embeddings=dense.tok_embeddings,
        norm=dense.norm,
        lm_head=dense.lm_head,
        layers=dense.layers[:trunk_depth],
        heads=dense.layers[trunk_depth:],
    )


def gloeckle_model_spec(
    shape_name: str,
    *,
    num_heads: int,
    vocab_size: int,
    seq_len: int,
    attn_backend: str = "flex",
) -> ModelSpec:
    return ModelSpec(
        name="mtp_titan_gloeckle",
        flavor=f"{shape_name}_n{num_heads}",
        model=gloeckle_model_config(
            shape_name,
            num_heads=num_heads,
            vocab_size=vocab_size,
            seq_len=seq_len,
            attn_backend=attn_backend,
        ),
        max_context_length=seq_len,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
    )
