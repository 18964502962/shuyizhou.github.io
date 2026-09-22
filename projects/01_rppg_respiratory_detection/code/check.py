
import json
import os

# 修改为你的PURE数据路径
pure_path = r"F:\data\PURE\01-01"
print("PURE 文件夹包含:", os.listdir(pure_path))

# 查看json内容
json_file = os.path.join(pure_path, "01-01.json")
with open(json_file, 'r') as f:
    data = json.load(f)
    print("PURE json 包含的字段:", list(data.keys()) if isinstance(data, dict) else "是列表格式")