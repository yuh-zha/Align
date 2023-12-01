from .inference import Inferencer
from typing import List, Tuple
import torch

class Align:
    def __init__(self, model: str, batch_size: int, device: int, ckpt_path: str, verbose=True) -> None:
        self.model = Inferencer(
            ckpt_path=ckpt_path, 
            model=model,
            batch_size=batch_size, 
            device=device,
            verbose=verbose
        )

    def predict(self, contexts: List[str], claims: List[str]) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Predict the alignment labels for context and claim pairs.

        Args:
            contexts (List[str]): A list of contexts.
            claims (List[str]): A list of claims.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]: A tuple of prediction scores `(regression_score, binary_score, nli_scores)`.
                Both `regression_score` and `binary_score` have shape (N,). `nli_scores` has shape (N, 3).
        """
        self.model.nlg_eval_mode = None
        return self.model.inference(contexts, claims)

    def score(self, contexts: List[str], claims: List[str], head: str = 'nli', split: bool = True) -> torch.FloatTensor:
        """Calculate the alignment scores between context and claim pairs using the specified prediction head and aggregation method.

        Args:
            contexts (List[str]): A list of contexts.
            claims (List[str]): A list of claims.
            head (str, optional): The prediction head to use ('nli', 'bin', or 'reg'). Defaults to 'nli'.
            split (bool, optional): Whether to split and then aggregate long inputs. If set to False, the model will trucate oversized inputs. Defaults to True.

        Returns:
            torch.FloatTensor: Alignment scores for the input pairs.
        """
        self.model.nlg_eval_mode = head + ('_sp' if split else '')
        return self.model.nlg_eval(contexts, claims)[1]