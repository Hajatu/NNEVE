from typing import Tuple

from pydantic import BaseModel


class QOParams(BaseModel):

    c: float

    class Config:
        allow_mutation = True
        arbitrary_types_allowed = True

    def update(self) -> None:
        self.c += 0.16

    def extra(self) -> Tuple[float, ...]:
        return (self.c,)
