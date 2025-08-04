import datasets, os
print("proxy ->", os.getenv("http_proxy"))
ds = datasets.load_dataset("cais/mmlu",'all' ,streaming=True)   # 只拉元数据，秒级完成
print("✓ Dataset fetched, first key:", list(ds.keys())[:3])