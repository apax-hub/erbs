from typing import Optional
import jax.numpy as jnp
from jax.ops import segment_sum


def segment_mean(data: jnp.ndarray,
                 segment_ids: jnp.ndarray,
                 num_segments: Optional[int] = None,
                 indices_are_sorted: bool = False,
                 unique_indices: bool = False):
  """Returns mean for each segment.
  Adapted from jraph
  https://github.com/google-deepmind/jraph/blob/51f5990104f7374492f8f3ea1cbc47feb411c69c/jraph/_src/utils.py#L85

  Args:
    data: the values which are averaged segment-wise.
    segment_ids: indices for the segments.
    num_segments: total number of segments.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted.
    unique_indices: whether ``segment_ids`` is known to be free of duplicates.
  """
  nominator = segment_sum(
      data,
      segment_ids,
      num_segments,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices)
  denominator = segment_sum(
      jnp.ones_like(data),
      segment_ids,
      num_segments,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices)
  return nominator / jnp.maximum(denominator,
                                 jnp.ones(shape=[], dtype=denominator.dtype))