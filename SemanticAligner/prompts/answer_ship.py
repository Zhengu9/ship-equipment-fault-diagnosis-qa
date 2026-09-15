# prompts/answer_ship.py

examples = [
    {
        "query": "大管轮在备车操作时，如果没对钟，评价标准是怎么要求的？",
        "evidences": [
            [
                ("大管轮", "[需考核科目]", "《动力装置测试分析与操作》"),
                ("《动力装置测试分析与操作》", "[包含评估任务]", "1.1 船舶电控柴油机的备车、起动、完车"),
                ("1.1 船舶电控柴油机的备车、起动、完车", "[评价标准]", "①接到驾驶台备车命令后，进行确认并与驾驶台联系对车钟、时钟；")
            ],
            [
                ("大管轮", "[需考核科目]", "《电气与自动控制》"),
                ("《电气与自动控制》", "[包含评估任务]", "5.1 常见传感器和执行阀件的故障诊断")
            ]
        ],
        "answer": "According to graph [1], the standard is '①接到驾驶台备车命令后，进行确认并与驾驶台联系对车钟、时钟；'. Graph [2] is not useful."
    },
    {
        "query": "轮机长的主机故障分析及其排除任务有哪些评估要素？",
        "evidences": [
            [
                ("轮机长", "[需考核科目]", "《轮机模拟器》"),
                ("《轮机模拟器》", "[包含评估任务]", "1.1 主机故障分析及其排除"),
                ("1.1 主机故障分析及其排除", "[评估要素]", "●1.1.1 主机故障分析及其排除①分析可能产生该故障的原因；②查找到设置的故障并排除。")
            ]
        ],
        "answer": "According to graph [1], the elements are '●1.1.1 主机故障分析及其排除①分析可能产生该故障的原因；②查找到设置的故障并排除。'"
    },
    {
        "query": "值班机工起动发电柴油机前的准备工作包括哪些要素？",
        "evidences": [
            [
                ("值班机工", "[需考核科目]", "《设备拆装与操作》"),
                ("《设备拆装与操作》", "[包含评估任务]", "1.1 发电柴油机的起动"),
                ("1.1 发电柴油机的起动", "[评估要素]", "◎1.1.2测量油底壳油位")
            ],
            [
                ("值班机工", "[需考核科目]", "《设备拆装与操作》"),
                ("《设备拆装与操作》", "[包含评估任务]", "1.1 发电柴油机的起动"),
                ("1.1 发电柴油机的起动", "[评估要素]", "◎1.1.10空气瓶放残水")
            ]
        ],
        "answer": "According to graphs [1][2], the elements include '◎1.1.2测量油底壳油位' and '◎1.1.10空气瓶放残水'."
    },
    {
        "query": "轮机长在主机故障排除后需要做什么？",
        "evidences": [
            [
                ("1.1 主机故障分析及其排除", "[评价标准]", "⑤故障排除后恢复主机至正常运行状态。")
            ],
            [
                ("轮机长", "[需考核科目]", "《机舱资源管理》"),
                ("《机舱资源管理》", "[包含评估任务]", "1.13 船舶接船管理")
            ]
        ],
        "answer": "According to graph [1], the requirement is '⑤故障排除后恢复主机至正常运行状态。'. Graph [2] is not useful."
    },
    {
        "query": "二/三管轮进行气缸盖拆装时有哪些准备工作标准？",
        "evidences": [
            [
                ("二/三管轮", "[需考核科目]", "《动力设备拆装》"),
                ("《动力设备拆装》", "[包含评估任务]", "1.1 气缸盖拆装与检查"),
                ("1.1 气缸盖拆装与检查", "[评价标准]", "①吊装工具及起重设备安全检查；")
            ],
            [
                ("二/三管轮", "[需考核科目]", "《动力设备拆装》"),
                ("《动力设备拆装》", "[包含评估任务]", "1.1 气缸盖拆装与检查"),
                ("1.1 气缸盖拆装与检查", "[评价标准]", "④穿戴劳保防护服（安全帽、工作鞋等）。")
            ]
        ],
        "answer": "According to graphs [1][2], the standards include '①吊装工具及起重设备安全检查；' and '④穿戴劳保防护服（安全帽、工作鞋等）。'"
    }
]


def get(query, evidences, shot=5):
    prompt = """Please answer the question based on the given evidences from a knowledge graph related to Marine Seafarer Competency Evaluation (轮机专业适任评估). 

Notes)

1). Use the original text in the valid evidences as answer output, NEVER rephrase or reformat the professional standards (e.g., keep symbols like ①, ◎, ●).
2). There may be different answers for different evidences. Return all possible answers for every evidence graph, except for those that are obviously not aligned with the maritime role or task in the query.
3). You should provide a brief reason with several words (e.g., "the standard is...", "the element is..."), then tell all the answers you found.
4). You are a professional Chief Engineer (轮机长) instructor. Be precise.

Examples)
"""

    for i in range(min(shot, len(examples))):
        prompt += f"""
{i+1}. query: '{examples[i]['query']}'
    evidences: {", ".join([f"graph [{j+1}]: {evidence}" for j, evidence in enumerate(examples[i]['evidences'])])}
    answer: '{examples[i]['answer']}'
"""

    return prompt + f"""
Your task)
**Read and follow the instructions and examples step by step**
query: '{query}'
evidences: {", ".join([f"graph [{j+1}]: {evidence}" for j, evidence in enumerate(evidences)])}
"""