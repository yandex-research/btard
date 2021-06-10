import os
import random
import sys
from enum import Enum
from typing import Any

from hivemind.utils.logging import get_logger


logger = get_logger(__name__)


class AttackType(Enum):
    NONE = 0
    SIGN_FLIPPING = 1
    LABEL_FLIPPING = 2
    CONSTANT_DIRECTION = 3


class AttackManager:
    def __init__(self, seed: Any):
        self._attack_type = AttackType[os.getenv('ATTACK_TYPE', 'NONE')]
        self._attack_start_step = int(os.getenv('ATTACK_START', 3000))
        self._max_attacks = int(os.getenv('MAX_ATTACKS', 10 ** 9))
        self._attack_proba = float(os.getenv('ATTACK_PROBA', 1.0))
        self._check_proba = float(os.getenv('CHECK_PROBA', 0.5 * 1 / 16))

        self._random = random.Random(seed)
        self._n_attacks = 0

    def should_attack(self, attack_type: AttackType, step: int) -> bool:
        result = (
            self._attack_type == attack_type and
            step >= self._attack_start_step and
            self._n_attacks < self._max_attacks and
            self._random.random() < self._attack_proba
        )
        if result:
            self._n_attacks += 1
            logger.warning(f'Attacking with {attack_type.name}')

            if self._random.random() < self._check_proba:
                logger.fatal('This peer has been banned')
                sys.exit(1)
        return result
