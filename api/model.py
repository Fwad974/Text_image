from typing import Optional

from pydantic import BaseModel


class Input(BaseModel):
    description: str