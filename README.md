# 模型下载

若在中国可考虑使用命令行

```bash
# MacOS or Linux
export HF_ENDPOINT="https://hf-mirror.com"
# Windows Powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 下载 roberta-base 到 pretrained 文件夹
huggingface-cli download FacebookAI/roberta-base --local-dir ./pretrained/roberta-base
```