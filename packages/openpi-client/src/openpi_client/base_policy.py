import abc
from typing import Dict, List, Sequence


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def infer_many(self, obs: Sequence[Dict]) -> List[Dict]:
        """Infer actions from multiple observations.

        Implementations may override this with a true batched forward pass. The
        default preserves existing behavior by evaluating requests one at a time.
        """
        return [self.infer(item) for item in obs]

    def warmup_many(self, obs: Dict, batch_sizes: Sequence[int]) -> None:
        """Warm up batched inference for the provided observation shape.

        Implementations may override this to compile/cache selected batch sizes
        without changing future policy outputs. The default is intentionally a
        no-op because generic policies may have stateful or non-idempotent
        inference.
        """
        del obs, batch_sizes

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
