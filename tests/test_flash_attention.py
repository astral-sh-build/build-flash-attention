import math
from importlib.metadata import version

import flash_attn
import pytest
import torch
from flash_attn import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
)


@pytest.fixture(scope="module")
def device() -> torch.device:
    assert torch.cuda.is_available(), "The tests must run on a CUDA GPU"
    return torch.device("cuda")


def attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
) -> torch.Tensor:
    key = key.repeat_interleave(query.shape[2] // key.shape[2], dim=2)
    value = value.repeat_interleave(query.shape[2] // value.shape[2], dim=2)

    scores = torch.einsum(
        "bqhd,bkhd->bhqk",
        query.float() / math.sqrt(query.shape[-1]),
        key.float(),
    )

    query_positions = (
        torch.arange(query.shape[1], device=query.device)[:, None]
        + key.shape[1]
        - query.shape[1]
    )
    key_positions = torch.arange(key.shape[1], device=key.device)[None, :]
    allowed = torch.ones(
        (query.shape[1], key.shape[1]),
        dtype=torch.bool,
        device=query.device,
    )

    if causal:
        allowed &= key_positions <= query_positions
    if window_size[0] >= 0:
        allowed &= key_positions >= query_positions - window_size[0]
    if window_size[1] >= 0:
        allowed &= key_positions <= query_positions + window_size[1]

    scores = scores.masked_fill(~allowed, float("-inf"))
    probabilities = scores.softmax(dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", probabilities, value.float())

    return output.to(query.dtype)


def assert_attention_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_published_cuda_wheel(device: torch.device) -> None:
    assert version("flash-attn") == "2.8.3+cu.12.8.torch.2.10"
    assert flash_attn.__version__ == "2.8.3"
    assert torch.__version__ == "2.10.0+cu128"
    assert torch.version.cuda == "12.8"
    assert torch.cuda.get_device_name(device)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attention(
    device: torch.device,
    causal: bool,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(0)
    query, key, value = (
        torch.randn((2, 16, 4, 32), device=device, dtype=dtype) for _ in range(3)
    )

    actual = flash_attn_func(query, key, value, causal=causal)
    expected = attention_reference(query, key, value, causal=causal)

    assert_attention_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attention_backward(device: torch.device, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    inputs = tuple(
        torch.randn(
            (2, 16, 4, 32),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        for _ in range(3)
    )
    reference_inputs = tuple(
        tensor.detach().clone().requires_grad_() for tensor in inputs
    )

    actual = flash_attn_func(*inputs, causal=True)
    expected = attention_reference(*reference_inputs, causal=True)
    gradient = torch.randn_like(actual)

    actual_gradients = torch.autograd.grad(actual, inputs, gradient)
    expected_gradients = torch.autograd.grad(
        expected,
        reference_inputs,
        gradient,
    )

    assert_attention_close(actual, expected)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=3e-2,
            rtol=3e-2,
        )


@pytest.mark.parametrize("causal", [False, True])
def test_qkvpacked_attention(device: torch.device, causal: bool) -> None:
    torch.manual_seed(0)
    qkv = torch.randn(
        (2, 16, 3, 4, 32),
        device=device,
        dtype=torch.float16,
        requires_grad=True,
    )

    actual = flash_attn_qkvpacked_func(qkv, causal=causal)
    expected = attention_reference(*qkv.unbind(dim=2), causal=causal)

    assert_attention_close(actual, expected)
    actual.square().mean().backward()

    assert qkv.grad is not None
    assert torch.isfinite(qkv.grad).all()


@pytest.mark.parametrize("causal", [False, True])
def test_kvpacked_attention(device: torch.device, causal: bool) -> None:
    torch.manual_seed(0)
    query = torch.randn(
        (2, 16, 4, 32),
        device=device,
        dtype=torch.float16,
        requires_grad=True,
    )
    key_value = torch.randn(
        (2, 16, 2, 2, 32),
        device=device,
        dtype=torch.float16,
        requires_grad=True,
    )

    actual = flash_attn_kvpacked_func(query, key_value, causal=causal)
    expected = attention_reference(
        query,
        *key_value.unbind(dim=2),
        causal=causal,
    )

    assert_attention_close(actual, expected)
    actual.square().mean().backward()

    assert query.grad is not None
    assert key_value.grad is not None
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(key_value.grad).all()


@pytest.mark.parametrize("causal", [False, True])
def test_grouped_query_attention(device: torch.device, causal: bool) -> None:
    torch.manual_seed(0)
    query = torch.randn((2, 16, 4, 32), device=device, dtype=torch.float16)
    key, value = (
        torch.randn((2, 16, 2, 32), device=device, dtype=torch.float16)
        for _ in range(2)
    )

    actual = flash_attn_func(query, key, value, causal=causal)
    expected = attention_reference(query, key, value, causal=causal)

    assert_attention_close(actual, expected)


@pytest.mark.parametrize("causal", [False, True])
def test_variable_length_attention(device: torch.device, causal: bool) -> None:
    torch.manual_seed(0)
    sequence_lengths = (5, 9, 3)
    boundaries = torch.tensor([0, 5, 14, 17], device=device, dtype=torch.int32)
    query, key, value = (
        torch.randn(
            (sum(sequence_lengths), 4, 32),
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )
        for _ in range(3)
    )

    actual = flash_attn_varlen_func(
        query,
        key,
        value,
        boundaries,
        boundaries,
        max(sequence_lengths),
        max(sequence_lengths),
        causal=causal,
    )
    expected = torch.cat(
        [
            attention_reference(
                query[start:end][None],
                key[start:end][None],
                value[start:end][None],
                causal=causal,
            )[0]
            for start, end in zip((0, 5, 14), (5, 14, 17), strict=True)
        ]
    )

    assert_attention_close(actual, expected)
    actual.square().mean().backward()

    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize("causal", [False, True])
def test_variable_length_qkvpacked_attention(
    device: torch.device,
    causal: bool,
) -> None:
    torch.manual_seed(0)
    sequence_lengths = (5, 9, 3)
    boundaries = torch.tensor([0, 5, 14, 17], device=device, dtype=torch.int32)
    qkv = torch.randn(
        (sum(sequence_lengths), 3, 4, 32),
        device=device,
        dtype=torch.float16,
        requires_grad=True,
    )

    actual = flash_attn_varlen_qkvpacked_func(
        qkv,
        boundaries,
        max(sequence_lengths),
        causal=causal,
    )
    expected = torch.cat(
        [
            attention_reference(
                *(tensor[None] for tensor in qkv[start:end].unbind(dim=1)),
                causal=causal,
            )[0]
            for start, end in zip((0, 5, 14), (5, 14, 17), strict=True)
        ]
    )

    assert_attention_close(actual, expected)
    actual.square().mean().backward()

    assert qkv.grad is not None
    assert torch.isfinite(qkv.grad).all()


@pytest.mark.parametrize("window_size", [(3, 0), (2, 2)])
def test_sliding_window_attention(
    device: torch.device,
    window_size: tuple[int, int],
) -> None:
    torch.manual_seed(0)
    query, key, value = (
        torch.randn((2, 16, 4, 32), device=device, dtype=torch.float16)
        for _ in range(3)
    )

    actual = flash_attn_func(query, key, value, window_size=window_size)
    expected = attention_reference(
        query,
        key,
        value,
        window_size=window_size,
    )

    assert_attention_close(actual, expected)
