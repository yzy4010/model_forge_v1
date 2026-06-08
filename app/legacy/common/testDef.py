from threading import Event as ThreadPoolEvent
import time,cv2,json,asyncio
from ajaxResult import *
from config import *

def file_processing_task(stop_event: ThreadPoolEvent, file_path: str):
    """模拟文件处理"""
    try:
        with open(file_path, 'w',encoding="utf-8") as f:
            for i in range(10):
                if stop_event.is_set():
                    print("检测到停止请求，中止文件写入")
                    return
                f.write(f"Line {i}\n")
                time.sleep(1)
        return {"lines_written": 10}
    except Exception as e:
        raise RuntimeError(f"文件处理失败: {str(e)}")
    
    
# print(asyncio.run(postAsyncJsonWithOutJwt(JAVA_API_PATH+"/tag/labelList/fromPython",{
#                                 "guId":"hahaha",
#                                 "imageName":"1111111111111_frames_00160.jpg",
#                                 "cocoJsonStr":"{\"info\":{\"description\":\"Video Inference Results\",\"fps\":30,\"width\":1920,\"height\":1080,\"license\":\"FORVIA\"},\"images\":[{\"id\":1,\"width\":1920,\"height\":1080,\"file_name\":\"1111111111111_frames_00160.jpg\",\"license\":1,\"date_captured\":\"\"}],\"annotations\":[{\"id\":1,\"image_id\":1,\"category_id\":6,\"segmentation\":[],\"bbox\":[657,498,999,1062],\"score\":0.25507789850234985,\"area\":192888,\"iscrowd\":0}],\"licenses\":[{\"id\":1,\"name\":\"FORVIA\",\"url\":\"\",\"file_url\":\"\",\"version\":\"1.0\"}],\"categories\":[{\"id\":1,\"name\":\"握笔\"},{\"id\":2,\"name\":\"压板\"},{\"id\":3,\"name\":\"盖盖子\"},{\"id\":4,\"name\":\"检查\"},{\"id\":5,\"name\":\"右手锤\"},{\"id\":6,\"name\":\"左手按\"},{\"id\":7,\"name\":\"离场\"},{\"id\":8,\"name\":\"插卡\"}]}"
#                                 })))
# postresults=asyncio.gather(task_to_java)
# print(postresults)
# print(cv2.getBuildInformation())
# frame_count=10
# DEDUCTION_FM_OUTPUT_IMAGE_PATH="/data/deduction/data/images"
# print(f"{DEDUCTION_FM_OUTPUT_IMAGE_PATH}/hahahahah_frames_{frame_count:05d}.jpg")