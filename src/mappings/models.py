from typing import List, Optional

from pydantic import BaseModel


class NAF2025(BaseModel):
    code: str
    label: str
    include: Optional[str] = None
    not_include: Optional[str] = None
    notes: Optional[str] = None


class NAF2008(BaseModel):
    code: str
    label: str
    naf2025: List[NAF2025] = []
