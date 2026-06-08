import json
import time
import requests, httpx, asyncio
from .config import *


def ajaxReturnYes(data: object, msg: str = "操作成功"):
    return {"code": 200, "msg": msg, "data": data}


def ajaxReturnNo(errCode: int, errMsg: str = "操作成功"):
    return {"code": errCode, "msg": errMsg, "data": None}


def postJsonWithOutJwt(url: str, jsonData: object):
    if url == JAVA_API_PATH + SEND_PROCESS_RESULT_TO_JAVA:   #如果是推送日志，延迟6秒
        time.sleep(6)
    try:
        # headers = {"Content-Type": "application/json"}
        response = requests.post(
            url,
            json=jsonData,  # 直接使用 json 参数会自动序列化
            timeout=30
            # 或者手动序列化：
            # data=json.dumps(json_data),
            # headers=headers
        )
        # print("post java status_code------>",response.status_code)
        # print("post java json------>",response.json())
        return response.json()
    except Exception as e:
        # print("post java error------>",e)   
        return json.dumps(ajaxReturnNo(500, "请求失败"), ensure_ascii=False)
    
def getJsonWithOutJwt(url: str,params: object=None):
    # time.sleep(1)
    try:
        # headers = {"Content-Type": "application/json"}
        response = requests.get(
            url,
            params=params,  # 直接使用 json 参数会自动序列化
            timeout=30
        # 或者手动序列化：
        # data=json.dumps(json_data),
        # headers=headers
        ) 
        # print("post java status_code------>",response.status_code)
        # print("post java json------>",response.json())
        return response.json()
    except Exception as e:
        # print("post java error------>",e)   
        return json.dumps(ajaxReturnNo(500, "请求失败"), ensure_ascii=False)


async def postByAsyncClient(client, url: str, jsonData: object):
    # headers = {"Content-Type": "application/json"}
    try:
        response = await client.post(
            url,
            json=jsonData,  # 直接使用 json 参数会自动序列化
            timeout=30
        )
        print("post java status_code------>", response.status_code)
        return response.json()
    except Exception as e:
        print("post java error------>",e)   
        return json.dumps(ajaxReturnNo(500, "请求失败"), ensure_ascii=False)


async def postAsyncJsonWithOutJwt(url: str, jsonData: object):
    # await asyncio.sleep(20)
    async with httpx.AsyncClient() as client:
        return await postByAsyncClient(client, url,jsonData)
    
async def AsyncioPostGather(url: str, jsonData: object):
    # asyncio.gather
    await asyncio.create_task(postAsyncJsonWithOutJwt(url, jsonData))
