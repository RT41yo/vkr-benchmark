"""Stage-7 only: matched text controls and detector dataset protocols.

Importing this package never reads or modifies frozen Stage-6 artifacts.
"""

from .protocol import cover_seed, grouped_split, secret_id_for_prompt
from .cover import sample_reference_token, generate_cover

__all__ = ["cover_seed", "grouped_split", "secret_id_for_prompt", "sample_reference_token", "generate_cover"]
