from typing import Iterable

# Distinct "nothing buffered" marker so a legitimately-yielded None is not mistaken
# for an empty peek buffer (None is a valid stream item).
_UNSET = object()


class IterWithPeek[T]:
    def __init__(self, it: Iterable[T]):
        self.it = iter(it)
        self.peeked: object = _UNSET

    def __iter__(self):
        return self

    def __next__(self) -> T:
        if self.peeked is not _UNSET:
            item = self.peeked
            self.peeked = _UNSET
            return item  # type: ignore[return-value]
        return next(self.it)

    def has_next(self) -> bool:
        if self.peeked is not _UNSET:
            return True

        try:
            self.peeked = next(self.it)
            return True
        except StopIteration:
            return False

    def peek(self) -> T:
        if self.peeked is _UNSET:
            self.peeked = next(self.it)
        return self.peeked  # type: ignore[return-value]
