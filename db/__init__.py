# db：数据存储层（postgres/ 连接池与台账表、milvus/ 向量库客户端）。
# 依赖方向单向：本包只依赖 psycopg / pymilvus，绝不反向 import 上层。
