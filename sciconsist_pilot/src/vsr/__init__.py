"""
Verification-Stratified Reward (VSR) — 核心模块

根据 claim 的可验证性特征，将其路由到三层验证机制：
  Layer 0 (Symbolic): 精确数值验证，不可 hack
  Layer 1 (Semi-Symbolic): 关系/趋势验证
  Layer 2 (Learned): 语义级 learned verifier

主要流程:
  1. ClaimExtractor 从 model response 提取 atomic claims
  2. VerifiabilityRouter 路由每个 claim 到对应层
  3. SymbolicVerifier / SemiSymbolicVerifier 执行符号验证
  4. VSRReward 聚合各层 reward 为最终分数
"""

from sciconsist_pilot.src.vsr.types import (
    AtomicClaim,
    VerifiabilityProfile,
    VerificationResult,
    TableRecord,
    StructuredTable,
    VerificationLayer,
    normalize_entity,
)
from sciconsist_pilot.src.vsr.claim_extractor import ClaimExtractor
from sciconsist_pilot.src.vsr.router import VerifiabilityRouter
from sciconsist_pilot.src.vsr.symbolic import SymbolicVerifier
from sciconsist_pilot.src.vsr.semi_symbolic import SemiSymbolicVerifier
from sciconsist_pilot.src.vsr.reward import VSRReward, VSRConfig
from sciconsist_pilot.src.vsr.table_index import TableIndex
from sciconsist_pilot.src.vsr.learned import FEHVerifier, FEHVerifierConfig

__all__ = [
    "AtomicClaim",
    "VerifiabilityProfile",
    "VerificationResult",
    "TableRecord",
    "StructuredTable",
    "VerificationLayer",
    "ClaimExtractor",
    "VerifiabilityRouter",
    "SymbolicVerifier",
    "SemiSymbolicVerifier",
    "VSRReward",
    "VSRConfig",
    "TableIndex",
    "FEHVerifier",
    "FEHVerifierConfig",
]
