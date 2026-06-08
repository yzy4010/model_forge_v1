from fastapi import APIRouter, FastAPI, Request, status
from fastapi.responses import JSONResponse

router = APIRouter()
#####  https://zhuanlan.zhihu.com/p/701898817

@router.get("/item/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}