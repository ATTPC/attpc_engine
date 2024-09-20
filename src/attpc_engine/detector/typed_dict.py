from typing import TypeVar, Generic, Iterable

K = TypeVar("K")
V = TypeVar("V")


class NumbaTypedDict(Generic[K, V]):
    """This is simply a type hint interface for Numba typed dictionaries

    Use this as a type hint wherever you would be using numba.typed.Dict
    and magically all of your linters will be happy.

    Example:
    ```py
    my_dict = numba.typed.Dict.empty(
        key_type=numba.core.types.int64,
        value_type=numba.core.types.int64
    )
    ```
    is type-hinted as
    ```py
    my_dict: NumbaTypedDict[int, int]
    ```

    Do not attempt to instantiate this object! It won't work!
    """

    def __getitem__(self, x: K) -> V: ...

    def __setitem__(self, x: K, v: V): ...

    def __len__(self) -> int: ...

    def items(self) -> Iterable[tuple[K, V]]: ...

    def get(self, x: K, default: V | None = None) -> V: ...
