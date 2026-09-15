#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import asyncio
import threading
from typing import Optional

ROOT = Path(__file__).resolve().parent

_bg_loop: Optional[asyncio.AbstractEventLoop] = None
_bg_thread: Optional[threading.Thread] = None
_bg_rag_Retriever: Dict[str, Any] = {}


def _start_bg_loop() -> asyncio.AbstractEventLoop:
    global _bg_loop, _bg_thread
    if _bg_loop is not None:
        return _bg_loop

    loop = asyncio.new_event_loop()

    def _run_loop():
        try:
            asyncio.set_event_loop(loop)
            loop.run_forever()
        finally:
            try:
                loop.close()
            except Exception:
                pass

    th = threading.Thread(target=_run_loop, name='rag-bg-loop', daemon=True)
    th.start()
    _bg_loop = loop
    _bg_thread = th
    return _bg_loop


def _create_rag_instance_sync(name: str, path_str: str, embed_model: str = 'BAAI/bge-m3', llm: str = 'qwen:7b'):
    from sentence_transformers import SentenceTransformer
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.utils import EmbeddingFunc

    st = None
    try:
        st = SentenceTransformer(embed_model, trust_remote_code=True)
    except Exception:
        st = None

    async def local_embed_func(texts):
        if isinstance(texts, str):
            texts = [texts]
        return st.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    async def llm_model_func(prompt, **kwargs):
        from subprocess import run
        proc = run(['ollama', 'run', llm], input=prompt, text=True, capture_output=True)
        return proc.stdout.strip() or proc.stderr.strip()

    embedding_dim = 1024 if st is not None and hasattr(st, 'get_sentence_embedding_dimension') and st.get_sentence_embedding_dimension() else 768
    embedding_func = EmbeddingFunc(embedding_dim=embedding_dim, max_token_size=8192, func=local_embed_func)

    config = RAGAnythingConfig(working_dir=str(path_str), parser='mineru', parse_method='auto', enable_image_processing=False, enable_table_processing=False, enable_equation_processing=False)
    rag = RAGAnything(config=config, llm_model_func=llm_model_func, embedding_func=embedding_func)
    return rag

MECH_KEYWORDS = [
    '主柴油机', '电控柴油机', '气缸盖', '气阀机构', '气缸套', '活塞组件', '活塞环', '连杆',
    '连杆大端轴瓦', '连杆螺栓', '主轴承', '喷油泵', '喷油器', '曲轴', '气缸起动阀', '安全阀',
    '示功阀', '空气分配器', '液压拉伸器', '增压器', '主机遥控系统', '主推进装置', '发电柴油机',
    '主发电机组', '应急发电机', '应急配电板', '主配电板', '岸电箱', '自动化电站', '辅锅炉',
    '锅炉燃烧器', '锅炉水位计', '分油机', '自清滤器', '空压机', '活塞式空气压缩机', '造水机',
    '制冷装置', '制冷压缩机', '空调装置', '舵机', '液压甲板机械', '液压泵', '柱塞泵', '液压马达',
    '液压控制阀', '离心泵', '往复泵', '齿轮泵', '换热器', '冷却器', '过滤器', '截止阀', '止回阀',
    '截止止回阀', '蝶阀', '压载水系统', '压载水处理装置', '舱底水系统', '油水分离器',
    '生活污水处理装置', '焚烧炉', '消防水系统', '蒸汽加温管路系统', '管系', '管路系统', '船舶电网',
    '高压配电装置', '高压电系统', '火灾探测系统', '火警系统', '风油切断装置', '报警监视系统',
    '机舱监视系统', '报警系统', '监测系统通信总线', '船舶同步发电机组', '螺旋桨轴', '螺旋桨轴辅助设备',

    '主柴油机常见故障', '辅助系统常见故障', '辅助机械设备常见故障', '螺旋桨轴故障', '辅助设备故障',
    '瘫船起动故障', '主发电机组起动故障', '主柴油机备车故障', '主柴油机起动故障', '机舱设备故障',
    '主柴油机应急故障', '全船失电故障', '发电机组并车故障', '舵机故障', '电机起动控制箱故障',
    'PLC 控制系统故障', '电气元件故障', '传感器故障', '执行阀件故障', '计算机控制系统故障',
    '监测系统故障', '接口功能模块故障', '编码器故障', '转换模块故障', '线路故障', '继电器板故障',
    '通信故障', '内存故障', 'CPU 死机', 'PT100 断线', '热电偶断开', '4-20mA 信号回路断开',
    '4-20mA 信号回路短路', '电动阀卡死', '气动阀漏气', '发电机外部短路故障', '发电机过载故障',
    '发电机失压故障', '发电机欠压故障', '船舶电网绝缘降低', '单相接地故障', '发电机主开关跳闸',
    '发电机起动失败', '非自动化电站主开关跳闸', '自动化电站主开关跳闸', '柴油机故障', '气阀故障',
    '气缸套磨损', '活塞磨损', '活塞销故障', '连杆小端轴承间隙异常', '活塞环故障', '曲轴销磨损',
    '主轴承间隙异常', '喷油泵故障', '供油定时异常', '喷油器故障', '启阀压力异常', '曲轴臂距差异常',
    '曲轴轴线异常', '增压器故障', '制冷装置故障', '液压系统故障', '泵浦故障', '管系泄漏', '管系堵塞',
    '锅炉故障', '分油机故障', '空调装置故障', '甲板机械故障', '油泵自动起动故障', '蓄电池故障',
    '船用电机故障', '电机接线故障', '电压互感器故障', '电流互感器故障', '照明设备故障',

    '气缸盖拆装', '气缸盖检查', '气阀机构拆装', '气阀机构检查', '气阀研磨', '气阀密封面检查',
    '气阀间隙测量', '气阀间隙调整', '气阀定时测量', '气阀定时调整', '气缸套拆装', '气缸套测量',
    '圆度计算', '圆柱度计算', '内径增大量计算', '活塞组件拆装', '活塞组件解体', '活塞测量',
    '活塞销检查', '连杆小端轴承间隙测量', '活塞环拆装', '活塞环检查', '活塞环天地间隙测量',
    '活塞环搭口间隙测量', '活塞环厚度测量', '活塞环槽测量', '连杆拆装', '连杆检查', '连杆大端轴瓦拆装',
    '连杆螺栓拆装', '连杆螺栓检查', '连杆螺栓上紧', '曲轴销测量', '主轴承拆装', '主轴承测量',
    '主轴承间隙测量', '喷油泵拆装', '喷油泵检修', '供油定时检查', '供油定时调整', '喷油泵密封性检查',
    '喷油泵密封性处理', '喷油器拆装', '喷油器检修', '启阀压力检查', '启阀压力调节', '曲轴臂距差测量',
    '曲轴臂距差计算', '曲轴轴线状态分析', '气缸起动阀拆装', '气缸起动阀检修', '安全阀拆装', '安全阀检修',
    '示功阀拆装', '示功阀检修', '空气分配器拆装', '空气分配器检修', '液压拉伸器使用', '液压拉伸器管理',
    '增压器拆卸', '增压器清洁', '增压器检查', '增压器测量', '增压器修理', '增压器装复', '制冷压缩机解体',
    '制冷压缩机清洁', '制冷压缩机修理', '制冷压缩机组装', '液压控制阀解体', '液压控制阀清洁', '液压控制阀修理',
    '液压控制阀组装', '液压泵解体', '液压泵清洁', '液压泵修理', '液压泵组装', '液压马达解体', '液压马达清洁',
    '液压马达修理', '液压马达组装', '自清滤器解体', '自清滤器检修', '自清滤器装复', '分油机解体', '分油机检修',
    '分油机装复', '离心泵拆卸', '离心泵清洗', '离心泵检查', '离心泵测量', '离心泵修理', '离心泵装复', '离心泵密封调整',
    '往复泵拆卸', '往复泵清洗', '往复泵检查', '往复泵测量', '往复泵修理', '往复泵装复', '往复泵密封调整',
    '齿轮泵拆卸', '齿轮泵清洗', '齿轮泵检查', '齿轮泵测量', '齿轮泵修理', '齿轮泵装复', '齿轮泵密封调整',
    '空压机拆卸', '空压机清洗', '空压机检查', '空压机测量', '空压机修理', '空压机组装', '锅炉水位计解体',
    '锅炉水位计清洁', '锅炉水位计修理', '锅炉水位计组装', '锅炉燃烧器解体', '锅炉燃烧器清洁', '锅炉燃烧器修理',
    '锅炉燃烧器组装', '双头螺栓安装', '螺栓安装', '阀门拆卸', '阀门清洗', '阀门检查', '阀门测量', '阀门修理',
    '阀门装复', '阀门试验', '换热器拆卸', '换热器清洗', '换热器检查', '换热器测量', '换热器修理', '换热器装复',
    '换热器试验', '管系拆装', '管系检查', '管系堵漏', '密封剂使用', '密封垫片使用', '密封填料使用', '机械装配',
    '焊接', '气焊', '气割', '手工电弧焊', '氧乙炔焊', '氧乙炔切割', '零部件清洁', '零部件检查', '零部件测量',
    '零部件修理', '零部件装复', '设备解体', '设备组装', '设备密封调整',

    '常用电气仪表', '万用表', '钳形表', '钳形电流表', '电压表', '电流表', '兆欧表', '便携式兆欧表', '安全用电',
    '电路符号', '电路图识读', '线路图识读', '控制箱元器件识别', '控制箱故障排查', '断线故障', '短路故障', '接地故障',
    '电子元器件识别', '电子控制线路图识读', '电路板焊接', '电子元器件装配', '自动空气断路器维护', '自动空气断路器故障排除',
    '热继电器', '继电器', '时间继电器', '温度继电器', '压力继电器', '电磁接触器', '熔断器', '塑壳断路器（MCCB）',
    '空气断路器（ACB）', '二极管', '三极管', '晶闸管', 'IGBT', 'PLC 模块', '电磁阀', '电动执行机构', '继电器参数整定',
    '接触器维护', '接触器参数整定', '船用电机', '船用电机拆装', '船用电机清洁', '船用电机轴承润滑脂添加', '船用电机故障判断',
    '船用电机接线', '电压互感器', '电流互感器', '照明设备维护', '蓄电池使用', '蓄电池维护', '智能传感器', '温度控制模块',
    '主机遥控系统操作', '发电机及配电系统操作', '辅锅炉控制系统操作', '分油机自动控制操作', '制冷和空调自动控制操作',
    '舵机控制操作', '泵和管系控制操作', '甲板机械电气控制操作', '电机起动控制', '油泵自动起动控制', '报警及监测系统',
    '变送器', '调节器', '数字式调节器', '调节器接线', '调节器操作', 'PLC 联机操作', 'PLC 程序上传', 'PLC 程序下载',
    'PLC 程序编辑', 'PLC 程序错误排查', '输入信号故障', '输出执行故障', '高压电检测', '高压电操作规程', '高压操作五防措施',
    '高压配电装置操作', '高压配电装置管理', '船舶电网', '船舶同步发电机', '应急配电板', '岸电箱', '自动化电站',
    '监测系统通信总线', '接口功能模块', '编码器', '转换模块', '继电器板', '计算机控制系统', 'CPU', '内存', '通信接口',
    '4-20mA 信号回路',

    '单元测试', '功能试验', '报警功能测试', '智能传感器测试', '温度控制模块测试', '主机遥控系统功能测试',
    '主机遥控系统故障处理', '发电机负载测试', '发电机保护功能测试', '发电机发动机保护功能测试', '船舶同步发电机组起动测试',
    '发电机组并车测试', '发电机组负荷转移测试', '发电机组解列测试', '发电机组功能试验', '船舶应急配电板功能试验',
    '发电机主开关功能测试', '辅锅炉控制系统保护功能测试', '辅锅炉控制系统故障处理', '分油机自动控制功能测试',
    '分油机自动控制故障处理', '制冷和空调自动控制保护功能测试', '制冷和空调自动控制故障处理', '舵机控制功能测试',
    '舵机控制故障处理', '泵和管系控制功能测试', '泵和管系控制故障处理', '甲板机械电气控制功能测试',
    '甲板机械电气控制故障处理', '油泵自动起动控制功能测试', '火灾探测系统功能测试', '变送器校准', '变送器调整',
    '自动控制系统单元功能测试', '测量单元性能测试', '调节单元性能测试', '执行阀件效能测试', '冷却水温度控制系统功能测试',
    '主推进装置安全保护功能测试', '副机安全保护功能测试', '燃油黏度自动控制系统功能测试', '辅锅炉安全保护功能测试',
    '辅锅炉自动控制系统功能测试', '分油机自动控制系统功能测试', '报警监视系统功能测试', '报警参数设置', '报警延时时间设置',
    '压力开关操作', '压力开关调整', '电动差压变送器操作', '电动差压变送器调整', '元器件功能测试', '电气元件功能测试',
    '执行机构功能测试',

    '船舶主柴油机备车', '主柴油机开航前备车准备', '主柴油机起动', '主柴油机起动后参数监测', '主柴油机起动后参数调整',
    '主柴油机定速后管理', '主柴油机完车操作', '主柴油机日常管理', '电控柴油机备车', '电控柴油机起动', '电控柴油机完车',
    '电控柴油机日常管理', '电控柴油机参数设定', '电控柴油机参数修改', '发电柴油机起动', '发电柴油机停车', '发电柴油机运行管理',
    '辅锅炉点火前准备', '辅锅炉点火', '辅锅炉升汽', '辅锅炉运行管理', '辅锅炉停炉操作', '分油机操作', '分油机运行管理',
    '空压机起动', '空压机停止', '空压机运行管理', '造水机操作', '造水机运行管理', '制冷装置起动', '制冷装置停用', '制冷装置日常管理',
    '制冷装置参数调整', '空调装置操作', '空调装置运行管理', '舵机起动', '舵机停止', '舵机系统日常管理', '舵机试验', '舵机调整',
    '液压甲板机械起动', '液压甲板机械停用', '液压系统日常管理', '液压甲板机械试验', '液压甲板机械调整', '离心泵启停操作',
    '离心泵工作性能判断', '压载水处理装置操作', '压载水处理装置管理', '舱底水系统操作', '舱底水系统管理', '蒸汽加温管路系统操作',
    '蒸汽加温管路系统管理', '生活污水处理装置操作', '焚烧炉操作', '焚烧炉运行管理', '油水分离器操作', '油水分离器运行管理',
    '船舶靠泊操作', '船舶离泊操作', '船舶加装燃润油料操作', '加油程序用语', '加油操作用语', '管路系统图识读', '设备运行管理',
    '设备日常维护保养', '设备参数监测', '设备参数调整',

    '瘫船起动操作', '海上瘫船状态应急操作', '应急发电机起动', '主发电机组起动', '机舱设备应急操作', '主柴油机应急操作',
    '全船失电应急操作', '发电机组并车故障应急操作', '舵机应急操作', '主机自动减速恢复程序', '主机自动停车恢复程序',
    '机动操作转换', '机动操作方法', '全船停电恢复程序', '副机重新起动', '备用副机起动', '电力供应恢复', '火警系统动作后故障排除',
    '火警系统动作后功能恢复', '风油切断装置动作后故障排除', '风油切断装置动作后功能恢复', '船舶 PSC 检查', '船舶 FSC 检查',
    '恶劣海况航行操作', '大风浪航行操作', '船舶搁浅应急处理', '加油溢油应急处理', '机舱火灾应急处理', '机舱进水应急处理',
    '船舶碰撞应急处理', '安全用电规程', '高压电安全操作', '管系堵漏应急处理', '船舶应急操作', '设备应急恢复', '安全防护措施',

    '手动工具', '动力工具', '钻床', '磨床', '普通车床', '手工电弧焊机', '氧乙炔气体设备', '测量仪器', '温度测量仪表',
    '压力测量仪表', '转速表', '功率表', '频率表', '温度表', '压力表', '电动差压变送器', '数字式调节器', 'Pt100', '热电偶',
    '热敏电阻', '光敏电阻', '光电池', '差动变压器', '磁感应接近开关', '量具', '密封工具', '焊接工具', '切割工具', '机械工程制图工具',

    '厂修管理', '船舶接船管理', '修船管理', '船舶能耗数据收集', '船舶能耗数据记录', '船舶能耗数据报告', '海事管理机构报告',
    '软件备份', '软件记录', '参数备份', '参数记录', '软件版本跟踪', '软件版本升级', '计算机应用程序编辑', '计算机应用程序保存',
    '船舶证书检查', '船舶合规管理', '设备维护记录', '设备检修记录',

]

def run_ollama(model: str, prompt: str, timeout: int = 30) -> str:
    try:
        proc = subprocess.run(['ollama', 'run', model], input=prompt, text=True, capture_output=True, timeout=timeout)
        out = proc.stdout.strip() or proc.stderr.strip()
        return out
    except Exception as e:
        return f"<ollama error: {e}>"


def classify_question(question: str, llm_model: str = 'qwen:7b') -> Dict[str, Any]:

    q = question.strip()
    q_lower = q.lower()
    for kw in MECH_KEYWORDS:
        if kw in q_lower:
            return {'route': 'GraphReasoner_SemanticAligner', 'complexity': 'short', 'confidence': 0.95, 'reason': f'keyword:{kw}', 'question': q}

    prompt_template = """
    【最高强制输出要求（必须100%遵守）】
1.  仅输出1个纯标准JSON对象，无任何其他内容（无解释、备注、代码块、符号、问候语）
2.  JSON仅含2个固定字段:"route","reason"，禁止增减/修改字段名、拼写错误
3.  严格使用双引号包裹键名和字符串值，JSON语法无错误，无多余换行/符号
4.  仅做路由分类，禁止解答/改写/补充用户问题，禁止输出与路由无关内容

【你的角色】
船舶轮机领域RAG系统专属路由分类器，仅基于用户问题、下方关键词库和路由规则，精准判定路由并输出合规JSON。

【核心路由规则（route仅能三选一，严格执行）】
1.  route="GraphReasoner_SemanticAligner": 用户问题包含下方关键词库任意1个及以上关键词，或核心围绕关键词库细分方向展开，用此值
2.  route="FullDocRetriever": 用户问题不含轮机关键词或属于领域外问题，用此值
3.  route="All": 复杂问题（多子问题或覆盖多类）用此值

【用户问题】
{user_question}

仅返回一个 JSON 对象，不要有多余文字，并且JSON对象前后不要有```这样的标识，严格遵守上述要求和规范。
"""

    prompt = prompt_template.replace('{user_question}', q)

    out = run_ollama(llm_model, prompt, timeout=120)
    #print(f"output: {out}")

    def _find_route(obj):
        if isinstance(obj, dict):
            if 'route' in obj:
                return obj['route'], obj.get('reason') if 'reason' in obj else None
            for v in obj.values():
                res = _find_route(v)
                if res is not None:
                    return res
        elif isinstance(obj, list):
            for item in obj:
                res = _find_route(item)
                if res is not None:
                    return res
        return None

    try:
        parsed = json.loads(out)
        if isinstance(parsed, dict) and 'route' in parsed:
            parsed.setdefault('question', q)
            return parsed
        found = _find_route(parsed)
        if found:
            route_val, reason_val = found
            result = {'route': route_val, 'reason': reason_val or 'nested', 'question': q}
            return result
    except Exception:
        pass
    import re
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", out, re.I)
    candidate = None
    if m:
        candidate = m.group(1)
    else:
        m2 = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", out)
        if m2:
            candidate = m2.group(1)

    if candidate:
        try:
            parsed2 = json.loads(candidate)
            if isinstance(parsed2, dict) and 'route' in parsed2:
                parsed2.setdefault('question', q)
                return parsed2
            found = _find_route(parsed2)
            if found:
                route_val, reason_val = found
                return {'route': route_val, 'reason': reason_val or 'nested', 'question': q}
        except Exception:
            pass

    return {'route': 'FullDocRetriever', 'complexity': 'short', 'confidence': 0.3, 'reason': 'parse_fail', 'question': q}


def call_adapter(path: Path, question: str, timeout: int = 1800) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        if 'FullDocRetriever' in str(path):
            try:
                import importlib
                init_rags = importlib.import_module('init_rags')
                RAG_Retriever = getattr(init_rags, 'RAG_Retriever', {})
                init_lightrag = getattr(init_rags, 'init_lightrag', None)
            except Exception as e:
                print('Failed importing init_rags for in-process FullDocRetriever:', e, file=sys.stderr)
                RAG_Retriever = {}

            if RAG_Retriever:
                bg_loop = _start_bg_loop()

                for key, info in RAG_Retriever.items():
                    if key in _bg_rag_Retriever:
                        continue
                    p_str = info.get('path') if isinstance(info, dict) else None
                    if not p_str:
                        continue
                    try:
                        async def _create_and_return(k, ps):
                            from sentence_transformers import SentenceTransformer
                            from raganything import RAGAnything, RAGAnythingConfig
                            from lightrag.utils import EmbeddingFunc

                            st = None
                            try:
                                st = SentenceTransformer('BAAI/bge-m3', trust_remote_code=True)
                            except Exception:
                                st = None

                            async def local_embed_func(texts):
                                if isinstance(texts, str):
                                    texts = [texts]
                                return st.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

                            async def llm_model_func(prompt, **kwargs):
                                from subprocess import run
                                proc = run(['ollama', 'run', 'qwen:7b'], input=prompt, text=True, capture_output=True)
                                return proc.stdout.strip() or proc.stderr.strip()

                            embedding_dim = 1024 if st is not None and hasattr(st, 'get_sentence_embedding_dimension') and st.get_sentence_embedding_dimension() else 768
                            embedding_func = EmbeddingFunc(embedding_dim=embedding_dim, max_token_size=8192, func=local_embed_func)

                            config = RAGAnythingConfig(working_dir=str(ps), parser='mineru', parse_method='auto', enable_image_processing=False, enable_table_processing=False, enable_equation_processing=False)
                            rag = RAGAnything(config=config, llm_model_func=llm_model_func, embedding_func=embedding_func)
                            return rag

                        fut = asyncio.run_coroutine_threadsafe(_create_and_return(key, p_str), bg_loop)
                        rag_obj = fut.result(timeout=300)
                        _bg_rag_Retriever[key] = {'path': p_str, 'rag': rag_obj}
                    except Exception as e:
                        print(json.dumps({'storage': str(p_str), 'error': 'bg_init_failed', 'detail': str(e)}, ensure_ascii=False), file=sys.stderr)
                        continue

                results_local = []
                for key, info in list(_bg_rag_Retriever.items()):
                    p = Path(info.get('path')) if info and info.get('path') else None
                    rag = info.get('rag') if info else None
                    if rag is None:
                        continue
                    try:
                        future = asyncio.run_coroutine_threadsafe(rag.aquery(question, mode='hybrid'), bg_loop)
                        try:
                            res = future.result(timeout=120)
                        except Exception as e_res:
                            print(json.dumps({'storage': str(p) if p is not None else key, 'error': 'aquery_failed', 'detail': str(e_res)}, ensure_ascii=False), file=sys.stderr)
                            continue

                        try:
                            if isinstance(res, dict):
                                text = res.get('answer') or res.get('text') or json.dumps(res, ensure_ascii=False)
                            else:
                                text = str(res)
                        except Exception:
                            text = str(res)

                        results_local.append({'text': text, 'score': 1.0, 'source': key})
                    except Exception as e:
                        print(json.dumps({'storage': str(p), 'error': 'aquery_failed', 'detail': str(e)}, ensure_ascii=False), file=sys.stderr)
                        continue

                return results_local

        proc = subprocess.run([sys.executable, str(path), '--question', question], input=None, text=True, capture_output=True, timeout=timeout)
        stdout_text = proc.stdout or ''
        stderr_text = proc.stderr or ''
        out = stdout_text.strip() or stderr_text.strip()
        if not out:
            return []
        try:
            data = json.loads(stdout_text)
        except Exception as e_json:
            import re
            combined = stdout_text + "\n" + stderr_text
            m = re.search(r"(\{.*\}|\[.*\])", combined, re.S)
            if m:
                candidate = m.group(1)
                try:
                    data = json.loads(candidate)
                except Exception:
                    print('Adapter JSON parse failed; raw output follows:', file=sys.stderr)
                    print('--- stdout ---', file=sys.stderr)
                    print(stdout_text, file=sys.stderr)
                    print('--- stderr ---', file=sys.stderr)
                    print(stderr_text, file=sys.stderr)
                    return []
            else:
                print('Adapter JSON parse failed; raw output follows:', file=sys.stderr)
                print('--- stdout ---', file=sys.stderr)
                print(stdout_text, file=sys.stderr)
                print('--- stderr ---', file=sys.stderr)
                print(stderr_text, file=sys.stderr)
                return []
        if isinstance(data, dict):
            raw = data.get('candidates', [])
        elif isinstance(data, list):
            raw = data
        else:
            return []

        normalized = []
        for item in raw:
            if isinstance(item, dict):
                normalized.append(item)
                continue
            if isinstance(item, (list, tuple)):
                text = str(item[0]) if len(item) > 0 else ''
                score = float(item[1]) if len(item) > 1 and isinstance(item[1], (int, float)) else 1.0
                source = str(item[2]) if len(item) > 2 else None
                normalized.append({'text': text, 'score': score, 'source': source})
                continue
            normalized.append({'text': str(item), 'score': 1.0, 'source': None})

        return normalized
    except Exception as e:
        print('Adapter call error', path, e)
        return []


def normalize_text(s: str) -> str:
    import re
    t = s.lower()
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r'[^0-9a-z\u4e00-\u9fff ]+', '', t)
    t = t.strip()
    return t


def merge_candidates(all_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = {}
    for c in all_candidates:
        text = c.get('text', None)
        if text is None:
            continue
        text = str(text).strip()
        if not text:
            continue
        cleaned_for_key = text.replace('*', '').replace(' ', '')
        norm = normalize_text(cleaned_for_key)
        if not norm:
            continue
        grp = groups.get(norm)
        if not grp:
            grp = {'norm': norm, 'texts': [text], 'sources': [c.get('source')], 'scores': [c.get('score', 1.0)], 'count': 1}
            groups[norm] = grp
        else:
            grp['texts'].append(text)
            grp['sources'].append(c.get('source'))
            grp['scores'].append(c.get('score', 1.0))
            grp['count'] += 1

    scored = []
    for g in groups.values():
        avg_score = sum(g['scores']) / len(g['scores']) if g['scores'] else 0
        unique_sources = len(set([s for s in g['sources'] if s]))
        weight = g['count'] + 0.1 * avg_score + 0.2 * unique_sources
        rep = max((t for t in g['texts'] if t), key=lambda x: len(x))
        rep_clean = rep.replace('*', '').replace(' ', '')
        scored.append({'text': rep_clean, 'weight': weight})

    scored.sort(key=lambda x: (-x['weight'], x['text']))

    ordered_texts = []
    seen = set()
    for item in scored:
        t = item.get('text')
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        ordered_texts.append(t)

    return ordered_texts


def save_history(entry: Dict[str, Any], path: Path = ROOT / 'QAhistory.jsonl'):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def export_history_to_xlsx(jsonl_path: Path, xlsx_path: Path) -> bool:
    try:
        rows = []
        if not jsonl_path.exists():
            print(f'历史文件未找到: {jsonl_path}', file=sys.stderr)
            return False
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                q = obj.get('question', '')
                a = obj.get('answer', '')
                rows.append({'question': q, 'answer': a})

        if not rows:
            rows = [{'question': '', 'answer': ''}]

        try:
            xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            df.to_excel(xlsx_path, index=False)
            return True
        except Exception:
            pass

        try:
            from openpyxl import Workbook
            wb = Workbook()
            ws = wb.active
            ws.append(['question', 'answer'])
            for r in rows:
                ws.append([r.get('question', ''), r.get('answer', '')])
            wb.save(xlsx_path)
            return True
        except Exception as e:
            print(f'导出失败: {e}', file=sys.stderr)
            return False
    except Exception as e:
        print(f'导出异常: {e}', file=sys.stderr)
        return False


def interactive_loop(llm_model: str = 'qwen:7b'):
    print('Interactive RAG orchestrator. Enter questions (quit/q to quit, output to export history).')
    adapters = {
        'GraphReasoner': ROOT / 'GraphReasoner' / 'scripts' / 'query_single.py',
        'SemanticAligner': ROOT / 'SemanticAligner' / 'pipeline' / 'query_single.py',
        'FullDocRetriever': ROOT / 'FullDocRetriever' / 'query_single.py'
    }

    while True:
        try:
            question = input('\n问题: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nExiting.')
            break
        if not question:
            break
        if question.lower() in ('q', 'quit'):
            print('退出。')
            break
        if question.lower() == 'output':
            jsonl_default = ROOT / 'QAhistory.jsonl'
            xlsx_default = ROOT / 'QAhistory.xlsx'
            print(f'导出历史记录: 默认 JSONL 路径 {jsonl_default}，默认导出文件 {xlsx_default}')
            dest = input('请输入导出 xlsx 的绝对路径（回车使用默认，输入 cancel 取消）: ').strip()
            if dest.lower() == 'cancel':
                print('已取消导出。')
                continue
            if not dest:
                xlsx_path = xlsx_default
            else:
                dest_p = Path(dest)
                if not dest_p.is_absolute():
                    print('请输入绝对路径（以 / 开头）。已取消。')
                    continue
                if dest_p.exists() and dest_p.is_dir():
                    xlsx_path = dest_p / 'QAhistory.xlsx'
                else:
                    if dest.endswith(os.sep):
                        xlsx_path = dest_p / 'QAhistory.xlsx'
                    else:
                        if dest_p.suffix.lower() != '.xlsx':
                            xlsx_path = dest_p.with_suffix('.xlsx')
                        else:
                            xlsx_path = dest_p

            ok = export_history_to_xlsx(jsonl_default, xlsx_path)
            if ok:
                print(f'已导出到 {xlsx_path}')
            else:
                print('导出失败，查看 stderr 输出以获取详细信息。')
            continue
        start = time.time()
        classification = classify_question(question, llm_model=llm_model)
        print('分类结果:', classification)

        route = classification.get('route', 'FullDocRetriever')
        complexity = classification.get('complexity', 'short')
        to_call = []
        if route == 'GraphReasoner_SemanticAligner':
            to_call = ['GraphReasoner']
        elif route == 'FullDocRetriever':
            to_call = ['FullDocRetriever']
        elif route == 'All' or complexity == 'complex' or classification.get('confidence', 1.0) < 0.4:
            to_call = ['GraphReasoner', 'SemanticAligner', 'FullDocRetriever']

        all_candidates = []
        for name in to_call:
            path = adapters.get(name)
            print(f"调用 {name} 适配器...")
            print(f"调用 SemanticAligner 适配器...")
            cand = call_adapter(path, question)
            for i, c in enumerate(cand):
                c.setdefault('source', name)
                c.setdefault('score', 1.0)
                all_candidates.append(c)

        merged_texts = merge_candidates(all_candidates)

        elapsed = time.time() - start
        merged_combined = "\n\n".join(merged_texts) if merged_texts else ''
        out = {
            'timestamp': time.time(),
            'question': question,
            'classification': classification,
            'raw_candidates': all_candidates,
            'merged_results': merged_combined,
            'answer': merged_combined,
            'elapsed_seconds': elapsed
        }

        print('\n=== 检索结果 ===')
        if merged_texts:
            print(merged_combined)
        else:
            print('(无候选)')

        save_history(out)
        print(f'已记录至 QAhistory.jsonl (耗时 {elapsed:.1f}s)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', help='run single query and exit')
    parser.add_argument('--init-rags', action='store_true', help='Initialize all RAG projects by calling init_rags.init_all', default=True)
    parser.add_argument('--llm', default='qwen:7b')
    args = parser.parse_args()

    if args.init_rags:
        try:
            print('Initializing all RAG projects via init_rags.init_all...')
            import init_rags
            init_rags.init_all(str(ROOT))
            print('init_rags.init_all completed.')
        except Exception as e:
            print(f'init_rags.init_all failed: {e}')

    if args.query:
        classification = classify_question(args.query, llm_model=args.llm)
        print('分类结果:', classification)
        question = args.query
        adapters = {
            'GraphReasoner': ROOT / 'GraphReasoner' / 'scripts' / 'query_single.py',
            'SemanticAligner': ROOT / 'SemanticAligner' / 'pipeline' / 'query_single.py',
            'FullDocRetriever': ROOT / 'FullDocRetriever' / 'query_single.py'
        }
        route = classification.get('route', 'FullDocRetriever')
        complexity = classification.get('complexity', 'short')
        to_call = []
        if route == 'GraphReasoner_SemanticAligner':
            to_call = ['GraphReasoner', 'SemanticAligner']
        elif route == 'FullDocRetriever':
            to_call = ['FullDocRetriever']
        else:
            to_call = ['GraphReasoner', 'SemanticAligner', 'FullDocRetriever']

        all_candidates = []
        for name in to_call:
            path = adapters.get(name)
            cand = call_adapter(path, question)
            for c in cand:
                c.setdefault('source', name)
                c.setdefault('score', 1.0)
                all_candidates.append(c)
        merged_texts = merge_candidates(all_candidates)
        merged_combined = "\n\n".join(merged_texts) if merged_texts else ''
        print(json.dumps({'classification': classification, 'merged': merged_combined, 'raw': all_candidates}, ensure_ascii=False, indent=2))
        save_history({'timestamp': time.time(), 'question': question, 'classification': classification, 'raw_candidates': all_candidates, 'merged': merged_combined, 'answer': merged_combined})
        return

    interactive_loop()


if __name__ == '__main__':
    main()
