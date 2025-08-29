import gradio as gr
from PIL import Image
from io import BytesIO
import requests

# -----------------------------
# Leader prompts for avatars
# -----------------------------
leaders = {
    "Marie Curie": "Full-body portrait of Marie Curie, realistic style, wearing early 1900s scientific attire",
    "Rosa Parks": "Full-body portrait of Rosa Parks, civil rights era clothing, realistic style",
    "Amelia Earhart": "Full-body portrait of Amelia Earhart, 1930s aviator outfit, realistic style",
    "Frida Kahlo": "Full-body portrait of Frida Kahlo, colorful artistic attire, realistic style",
    "Jane Austen": "Full-body portrait of Jane Austen, early 19th century dress, realistic style",
    "Malala Yousafzai": "Full-body portrait of Malala Yousafzai, modern clothing, realistic style",
    "Ada Lovelace": "Full-body portrait of Ada Lovelace, Victorian-era attire, realistic style",
    "Susan B. Anthony": "Full-body portrait of Susan B. Anthony, 19th century dress, realistic style",
    "Hatshepsut": "Full-body portrait of Hatshepsut, ancient Egyptian royal attire, realistic style",
    "Eleanor Roosevelt": "Full-body portrait of Eleanor Roosevelt, 1940s formal attire, realistic style",
    "Michelle Obama": "Full-body portrait of Michelle Obama, modern formal attire, realistic style"
}

# Hugging Face API
HF_API_TOKEN = ""  # Optional free token
HF_IMAGE_MODEL = "stabilityai/stable-diffusion-2"  # Free-to-use Stable Diffusion model
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

# -----------------------------
# Generate avatars dynamically
# -----------------------------
def generate_avatar(prompt):
    try:
        payload = {"inputs": prompt}
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}",
            headers=headers,
            json=payload,
            timeout=60
        )
        image_bytes = response.content
        return Image.open(BytesIO(image_bytes))
    except Exception as e:
        # Fallback placeholder if image generation fails
        img = Image.new("RGB", (256, 512), color=(200,200,200))
        return img

# Generate all avatars at startup
avatars = {name: generate_avatar(prompt) for name, prompt in leaders.items()}

# -----------------------------
# Hugging Face LLM for chat
# -----------------------------
HF_LLM_MODEL = "TheBloke/gpt4all-lora-quantized"  # Free model
def query_llm(prompt):
    try:
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
        response = requests.post(f"https://api-inference.huggingface.co/models/{HF_LLM_MODEL}",
                                 headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        outputs = response.json()
        return outputs[0]["generated_text"]
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

# -----------------------------
# Gradio chatbot function
# -----------------------------
def chat_with_leader(user_input, selected_leader, chat_history):
    leader_prompt = f"You are {selected_leader}. Give leadership and financial literacy advice to a student. Student asks: {user_input}"
    answer = query_llm(leader_prompt)
    
    chat_history = chat_history or []
    chat_history.append((f"You: {user_input}", f"{selected_leader}: {answer}"))
    
    avatar_img = avatars[selected_leader]
    return chat_history, avatar_img

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🌟 Her Story: AI Women's Leadership Platform")
    
    with gr.Row():
        selected_leader = gr.Dropdown(list(leaders.keys()), label="Choose a leader")
    
    chatbot = gr.Chatbot()
    avatar_image = gr.Image(label="Leader Avatar")
    user_input = gr.Textbox(label="Ask a question", placeholder="Type your question here...")
    
    user_input.submit(chat_with_leader, [user_input, selected_leader, chatbot], [chatbot, avatar_image])

demo.launch()
