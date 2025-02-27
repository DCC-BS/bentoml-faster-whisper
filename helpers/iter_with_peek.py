from typing import Iterable, TypeVar

T = TypeVar("T")


class IterWithPeek[T]:
    def __init__(self, it: Iterable[T]):
        self.it = iter(it)
        self.peeked = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.peeked is not None:
            item = self.peeked
            self.peeked = None
            return item
        return next(self.it)

    def has_next(self):
        if self.peeked is not None:
            return True

        try:
            self.peeked = next(self.it)
            return True
        except StopIteration:
            return False

    def peek(self):
        if self.peeked is None:
            self.peeked = next(self.it)
        return self.peeked
