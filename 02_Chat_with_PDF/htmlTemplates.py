import base64
from io import BytesIO
from pathlib import Path

from PIL import Image

IMAGES_DIR = Path(__file__).parent / "images"


def avatar_data_uri(file_name: str, size: int = 156) -> str:
    """Shrink a local image and inline it, so the avatars don't depend on an image host."""
    image = Image.open(IMAGES_DIR / file_name)
    image.thumbnail((size, size))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="{{AVATAR}}">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''.replace("{{AVATAR}}", avatar_data_uri("agent.png"))

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="{{AVATAR}}">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''.replace("{{AVATAR}}", avatar_data_uri("user.png"))
