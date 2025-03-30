import json
import random
import os

# 输入输出路径
input_path = 'utterances.jsonl'
output_dir = './data/parliament_subset/'

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# Step 1: 读取全部 utterances 文本内容
utterances = []
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        text = data.get('text') or data.get('utterance')
        if text and len(text.strip()) > 0:
            utterances.append(text.strip())

print(f"总共读取了 {len(utterances)} 条文本")

# Step 2: 打乱顺序并抽取 9000 条
random.shuffle(utterances)
subset = utterances[:9000]

# Step 3: 划分为 train / valid / test
train = subset[:7000]
valid = subset[7000:8000]
test = subset[8000:]

# Step 4: 写入三个文件
def write_list(filename, lines):
    with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')

write_list('train.txt', train)
write_list('valid.txt', valid)
write_list('test.txt', test)

print("✅ 生成完成！输出文件包括：")
print(f"  - {output_dir}train.txt ({len(train)} 行)")
print(f"  - {output_dir}valid.txt ({len(valid)} 行)")
print(f"  - {output_dir}test.txt  ({len(test)} 行)")