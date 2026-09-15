import sys
sys.path.append('..')

import json
from src.dataset import ShipQA
from src.indexer import Indexer

configs = json.load(open('../configs/ShipQA.json'))

dataset = ShipQA(configs)
KG = dataset.get_KG()

indexer = Indexer(configs)
print("正在连接 Milvus 并重置旧数据...")
try:

    indexer.node_vector_store.reset()
    indexer.relation_vector_store.reset()
    indexer.type_vector_store.reset()
    print(" 数据库重置成功！")
except AttributeError:
    print(" 自动重置失败：请检查 indexer.py 中 VectorStore 的变量名。")

print("开始构建索引，这可能需要几分钟...")
indexer.build_index(KG)
print(" 索引构建完成！")