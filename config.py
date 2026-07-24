import os
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

VAULT_API  = "https://127.0.0.1:27124"
API_KEY    = os.getenv("OBSIDIAN_API_KEY")
HEADERS    = {"Authorization": f"Bearer {API_KEY}"}
VERIFY_SSL = False
MODEL_NAME = "qwen2.5:7b"