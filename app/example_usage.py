import json
from rule_engine import RuleParser, RuleEngine

# --- 1. 定义全场景规则配置 ---
config = {
    "rule_config": [
        {
            "rule_id": "R1_BASIC_AND",
            "name": "基础逻辑：危险区域且抽烟",
            "enabled": True,
            "conditions": {
                "all": [
                    {"alias": "smoking"},
                    {"roi": "danger_zone"}
                ]
            },
            "action": {"type": "alert", "params": {"level": "high", "message": "发现危险区域吸烟"}}
        },
        {
            "rule_id": "R2_NESTED_COMPLEX",
            "name": "复杂逻辑：(抽烟 AND 手机) OR 危险区域",
            "enabled": True,
            "conditions": {
                "any": [
                    {
                        "all": [
                            {"alias": "smoking"},
                            {"alias": "calling"}
                        ]
                    },
                    {"roi": "danger_zone"}
                ]
            },
            "action": {"type": "alert", "params": {"level": "critical", "message": "严重违规操作"}}
        },
        {
            "rule_id": "R3_NOT_LOGIC",
            "name": "否定逻辑：抽烟且不是内部员工",
            "enabled": True,
            "conditions": {
                "all": [
                    {"alias": "smoking"},
                    {"not": {"alias": "is_staff"}}
                ]
            },
            "action": {"type": "alert", "params": {"level": "medium", "message": "外部人员吸烟"}}
        }
    ]
}


def run_comprehensive_test():
    print("🚀 [系统启动] 正在加载全场景测试用例...")

    # 初始化引擎
    rules = [RuleParser.parse_rule(r) for r in config["rule_config"] if r.get("enabled")]
    engine = RuleEngine(rules)

    # --- 测试矩阵 ---
    test_cases = [
        {
            "desc": "场景 A：危险区域 + 抽烟 (预期触发 R1, R2)",
            "data": {"smoking": True, "roi_tags": ["danger_zone"]}
        },
        {
            "desc": "场景 B：非危险区域 + 抽烟 + 打电话 (预期触发 R2, R3)",
            "data": {"smoking": True, "calling": True, "roi_tags": ["safe_zone"], "is_staff": False}
        },
        {
            "desc": "场景 C：抽烟 + 是内部员工 (预期 R3 不触发)",
            "data": {"smoking": True, "is_staff": True}
        },
        {
            "desc": "场景 D：边缘测试 - 空数据 (预期均不触发，且不报错)",
            "data": {}
        },
        {
            "desc": "场景 E：边缘测试 - 字段缺失 (预期不报错)",
            "data": {"something_else": True}
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n--- 测试序号 {i}: {case['desc']} ---")
        engine.evaluate_frame(case['data'])


if __name__ == "__main__":
    run_comprehensive_test()