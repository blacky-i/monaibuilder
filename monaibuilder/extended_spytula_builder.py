# Copyright (c) 2023 Chernenkiy Ivan, Sechenov University

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List

from spytula.builder import SpytulaBuilder


class ExtendedSpytulaBuilder(SpytulaBuilder):
    def __init__(self, *vars, **kwargs):
        super().__init__(*vars, **kwargs)

    @staticmethod
    def from_super(builder: SpytulaBuilder) -> ExtendedSpytulaBuilder:
        extended_builder = ExtendedSpytulaBuilder()
        extended_builder._key_format = builder._key_format
        extended_builder._data = builder._data
        extended_builder._root = builder._root
        return extended_builder

    def is_exist_node(self, key: str) -> bool:
        return key in self._data.keys()

    def _new_node(self) -> ExtendedSpytulaBuilder:
        """
        Helper method to create a new ExtendedSpytulaBuilder instance.

        Returns:
            ExtendedSpytulaBuilder: New instance of ExtendedSpytulaBuilder.
        """
        return type(self)()

    @contextmanager
    def node(self, key: str) -> Generator[ExtendedSpytulaBuilder, None, None]:
        """
        If key is not in builder, then creates new node
        """
        if not self.is_exist_node(key):
            with super().node(key) as new_element:
                yield new_element
        else:
            old_node = self._new_node()
            old_node._data = self._data[key]
            yield old_node

    @contextmanager
    def nodes(
        self, key: str
    ) -> Generator[Callable[[], ExtendedSpytulaBuilder], None, None]:
        """
        If key is not in builder, then creates new node
        """
        if not self.is_exist_node(key):
            with super().nodes(key) as new_element:
                yield new_element  # ExtendedSpytulaBuilder.from_super(new_element())
        else:
            if isinstance(self._data[key], list):
                old_nodes: List[Dict[str, Any]] = self._data[key]
                yield lambda: self.add_node(old_nodes)
            else:
                with super().nodes(key) as new_element:
                    yield new_element  # ExtendedSpytulaBuilder.from_super(new_element)
