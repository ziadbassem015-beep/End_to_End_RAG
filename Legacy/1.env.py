# %%
from dotenv import load_dotenv
import os
from pathlib import Path
env_path = Path(__file__).parent / "data" / "1.env"  
load_dotenv(env_path)

model = os.getenv("QWEN_MODEL")
print(f"QWEN_MODEL:{model}")
# %%
