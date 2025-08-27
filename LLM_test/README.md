如果你想在自己的服务器上进行这一步，需要自行部署大语言模型，我使用的Qwen2.5-7B-Instruct，这里有可以直接供你部署的代码download.py,以及api.py供你调试。
If you want to perform this step on your own server, you will need to deploy a large language model yourself. I used Qwen2.5-7B-Instruct. Here, I have provided download.py which you can use directly for deployment, and api.py for you to test the setup.
material.py文件是为了处理对话文本并整理出孔明的聊天对供微调使用，一起就绪后可以通过fintune.py文件调试，fintuning_test.py文件则供你查看最后效果。
The material.py file is used to process the dialogue text and organize Zhuge Liang's conversational pairs for fine-tuning. Once everything is ready, you can start the fine-tuning process using the fintune.py file. The fintuning_test.py file is then used to review the final results.
