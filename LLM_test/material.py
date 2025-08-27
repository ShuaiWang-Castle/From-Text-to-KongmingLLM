import json

def create_kongming_qa_dataset(file_path):
    """
    从三国演义的jsonl文件中提取孔明的问答对。

    这个函数会读取一个jsonl文件，识别出孔明作为回答者的对话场景，
    并将提问者的话作为"instruction"，将孔明的连续回复合并为"output"。

    Args:
        file_path (str): 输入的jsonl文件路径。

    Returns:
        list: 一个包含问答对字典的列表。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    dialogues = []
    for line in lines:
        # 移除可能存在的 source 标记
        if '{"role":' in line:
            clean_line = line[line.find('{"role":'):]
            try:
                dialogues.append(json.loads(clean_line))
            except json.JSONDecodeError:
                print(f"警告：无法解析行：{line.strip()}")
                continue

    qa_pairs = []
    i = 0
    while i < len(dialogues):
        # 寻找一个不由孔明发起的对话作为"instruction"
        if dialogues[i]['role'] != '孔明' and (i + 1) < len(dialogues):
            # 检查下一句是否是孔明的回复
            if dialogues[i+1]['role'] == '孔明':
                instruction = dialogues[i]['dialogue']
                
                # 开始合并孔明的连续回复
                output = ""
                j = i + 1
                while j < len(dialogues) and dialogues[j]['role'] == '孔明':
                    # 合并当前孔明的对话
                    output += dialogues[j]['dialogue']
                    
                    # 预读下一句，判断是否还需要继续合并
                    if (j + 1) < len(dialogues) and dialogues[j+1]['role'] != '孔明':
                        # 如果下一句是别人的话，就停止合并
                        break
                    elif (j + 1) < len(dialogues) and dialogues[j+1]['role'] == '孔明':
                         # 如果下一句还是孔明的话，检查再下一句是否是别人，
                         # 这是为了处理像您例子中提到的那种情况，即孔明连续说了几句话，但只有前两句是直接回复。
                         # 在这个场景下，我们假设连续的对话都是一个完整的回复，直到被另一个角色打断。
                         # 这个逻辑可以根据更复杂的规则进行调整。
                        j += 1
                    else:
                        # 到达文件末尾
                        break

                qa_pairs.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output.strip()
                })
                
                # 更新主循环的索引，跳过已经处理过的对话
                i = j
        i += 1
        
    return qa_pairs

# --- 主程序 ---
# 假设您的jsonl文件名为'三国演义.jsonl'
input_file = '/root/三国演义.jsonl'
output_file = '/root/孔明/json'

# 生成问答数据
kongming_qa_data = create_kongming_qa_dataset(input_file)

# 保存为json文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(kongming_qa_data, f, ensure_ascii=False, indent=4)

print(f"处理完成！数据已保存到 {output_file}")
print(f"共生成 {len(kongming_qa_data)} 条问答对。")