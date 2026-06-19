import httpx
import gradio as gr

CHAT_ENDPOINT = "http://localhost:8000/chat"

HEADER_HTML = """
<div style="text-align:center; padding:24px 0 16px;">
    <div style="font-size:1.5em; font-weight:700; margin-bottom:6px;">👋 Hi, I'm Osas's assistant</div>
    <div style="color:#6b7280; font-size:.9em;">Ask me anything about Osas — skills, projects, or experience.</div>
</div>
"""


def chat_fn(message: str, history: list) -> str:
    openai_history = [
        {"role": item["role"], "content": item["content"]}
        for item in history
    ]
    response = httpx.post(
        CHAT_ENDPOINT,
        json={"message": message, "history": openai_history},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["reply"]


with gr.Blocks(title="Osas's Portfolio Assistant") as demo:
    gr.HTML(HEADER_HTML)
    gr.ChatInterface(
        fn=chat_fn,
        chatbot=gr.Chatbot(
            height=460,
            placeholder="✨ What would you like to know about Osas?",
            show_label=False,
        ),
        textbox=gr.Textbox(
            placeholder="Ask about skills, projects, experience...",
            container=False,
            scale=7,
            show_label=False,
            lines=1,
            submit_btn="Send",
        ),
        examples=[
            "What are Osas's main technical skills?",
            "Tell me about his projects.",
            "What backend technologies does he work with?",
            "What's his experience with Java SDKs?",
        ],
        cache_examples=False,
    )


if __name__ == "__main__":
    demo.launch()
