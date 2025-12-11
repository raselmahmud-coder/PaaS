import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 初始化客户端
client = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),  # 请确保设置了环境变量
    base_url="https://api.moonshot.cn/v1",
)

# 发起聊天完成请求
try:
    response = client.chat.completions.create(
        model="kimi-k2-0905-preview",
        messages=[{"role": "system", "content": "You are a helpful assistant."}],
        temperature=0.3,
        max_tokens=8192,
        top_p=1,
        stream=True,
    )

    # 处理流式响应
    for chunk in response:
        choice = chunk.choices[0]
        if choice.delta and hasattr(choice.delta, "reasoning_content"):
            reasoning_content = getattr(choice.delta, "reasoning_content", None)
            if reasoning_content:
                print(reasoning_content, end="")
        if choice.delta and choice.delta.content is not None:
            print(choice.delta.content, end="")
    print()  # 换行

except Exception as e:
    print(f"请求失败: {e}")
