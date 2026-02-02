from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.api.schemas.infer import InferStreamRequest  # noqa: E402


def main() -> None:
    payload = {
        "scenario": {
            "scenario_id": "smoking_only",
            "models": [
                {
                    "alias": "smoking",
                    "model_id": "m_7576918032fa4dffa3c61ac0d6b25955",
                    "weights_path": "D:/projects/model_forge_v1/outputs/.../artifacts/best.pt",
                    "labels": ["smoke"],
                    "params": {
                        "conf": 0.35,
                        "iou": 0.45,
                        "imgsz": 640,
                        "max_det": 50,
                    },
                }
            ],
        },
        "rtsp_url": "rtsp://127.0.0.1:8554/mystream",
        "sample_fps": 2.0,
    }

    request = InferStreamRequest.model_validate(payload)

    assert request.rtsp_url == payload["rtsp_url"]
    assert request.sample_fps == 2.0
    assert request.scenario.scenario_id == payload["scenario"]["scenario_id"]
    assert request.scenario.models[0].alias == "smoking"
    assert request.scenario.models[0].params.imgsz == 640
    assert request.scenario.models[0].params.max_det == 50


if __name__ == "__main__":
    main()
