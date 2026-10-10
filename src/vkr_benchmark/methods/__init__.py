"""Steganographic method adapter contracts and implementations."""

from vkr_benchmark.methods.base import DecoderSession, EncoderSession, StegoMethod
from vkr_benchmark.methods.adg import (
    ADGConfig,
    ADGDecoderSession,
    ADGEncoderSession,
    ADGMethod,
)
from vkr_benchmark.methods.arithmetic import (
    ArithmeticConfig,
    ArithmeticDecoderSession,
    ArithmeticEncoderSession,
    ArithmeticMethod,
    ArithmeticPartition,
)
from vkr_benchmark.methods.bins import (
    BinsConfig,
    BinsDecoderSession,
    BinsEncoderSession,
    BinsMethod,
    BinsPartition,
)
from vkr_benchmark.methods.dair import (
    DAIRConfig,
    DAIRDecoderSession,
    DAIREncoderSession,
    DAIRMethod,
)
from vkr_benchmark.methods.discop import (
    DiscopConfig,
    DiscopDecoderSession,
    DiscopEncoderSession,
    DiscopMethod,
)
from vkr_benchmark.methods.rrc import (
    RRCConfig,
    RRCDecoderSession,
    RRCEncoderSession,
    RRCMethod,
    RRCPrecisionError,
    required_precision as rrc_required_precision,
)
from vkr_benchmark.methods.huffman import (
    HuffmanCodebook,
    HuffmanConfig,
    HuffmanDecoderSession,
    HuffmanEncoderSession,
    HuffmanMethod,
)
from vkr_benchmark.methods.types import (
    DecodeProgress,
    DecoderFinalization,
    EncodeDecision,
    EncoderFinalization,
    MethodEnvironment,
)

__all__ = [
    "ADGConfig",
    "ADGDecoderSession",
    "ADGEncoderSession",
    "ADGMethod",
    "ArithmeticConfig",
    "ArithmeticDecoderSession",
    "ArithmeticEncoderSession",
    "ArithmeticMethod",
    "ArithmeticPartition",
    "BinsConfig",
    "BinsDecoderSession",
    "BinsEncoderSession",
    "BinsMethod",
    "BinsPartition",
    "DAIRConfig",
    "DAIRDecoderSession",
    "DAIREncoderSession",
    "DAIRMethod",
    "DecodeProgress",
    "DecoderFinalization",
    "DecoderSession",
    "EncodeDecision",
    "EncoderFinalization",
    "EncoderSession",
    "DiscopConfig",
    "DiscopDecoderSession",
    "DiscopEncoderSession",
    "DiscopMethod",
    "HuffmanCodebook",
    "HuffmanConfig",
    "HuffmanDecoderSession",
    "HuffmanEncoderSession",
    "HuffmanMethod",
    "MethodEnvironment",
    "RRCConfig",
    "RRCDecoderSession",
    "RRCEncoderSession",
    "RRCMethod",
    "RRCPrecisionError",
    "rrc_required_precision",
    "StegoMethod",
]
