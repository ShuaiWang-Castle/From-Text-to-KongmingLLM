from langchainset import Qwen2_5_LLM
llm = Qwen2_5_LLM(mode_name_or_path = "/root/autodl-tmp/qwen/Qwen2.5-7B-Instruct")

print(llm("你是谁"))