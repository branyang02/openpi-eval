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

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
