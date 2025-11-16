"""
对配置进行哈希处理的模块

哈希链，前面阶段的配置一旦发生修改，后面阶段的哈希值都会发生变化
"""

from stablehash import stablehash


class ConfigHasher:

    def __init__(self, algorithm: str = "sha1"):
        """
        初始化 ConfigHasher

        :param algorithm: 哈希算法，默认为 "sha1"
        """
        self._algorithm = algorithm
        self._current_hash = ""

    def chain_hash(self, config: dict, inplace: bool = True) -> str:
        """
        计算配置字典在当前链条上的哈希值

        :param config: 配置字典
        :param inplace: 是否更新当前链尾的哈希值
        :return: 当前链尾的哈希值十六进制字符串
        """
        config_hash = stablehash(config, algorithm=self._algorithm).hexdigest()
        new_hash = stablehash(
            self._current_hash + config_hash, algorithm=self._algorithm
        ).hexdigest()
        if inplace:
            self._current_hash = new_hash
        return new_hash

    @property
    def current(self) -> str:
        """
        当前链尾的哈希值十六进制字符串

        :return: 当前链尾的哈希值十六进制字符串
        """
        return self._current_hash
