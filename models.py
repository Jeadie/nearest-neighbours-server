from typing import List, Optional, Union
import hnswlib
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel


class IndexParams(BaseModel):
    space: str
    dim: int
    max_elements: int
    M: Optional[int] = 16
    ef_construction: Optional[int] = 200
    random_seed: Optional[int] = 100
    allow_replace_deleted: Optional[bool] = False

class Item(BaseModel):
    id: int
    data: List[Union[float, int]]

class AddItemsParams(BaseModel):
    items: List[Item]
    num_threads: Optional[int] = -1
    replace_deleted: Optional[bool] = False


class KnnQueryParams(BaseModel):
    data: List[List[Union[float, int]]]
    k: Optional[int] = 1
    num_threads: Optional[int] = -1
    filter: Optional[List[int]] = None
