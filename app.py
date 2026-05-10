import streamlit as st
from groq import Groq
import re
import PyPDF2
import io
from PIL import Image
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json

import os
API_KEY = os.environ.get("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY", ""))
client = Groq(api_key=API_KEY)

JEE_PROMPT = """
You are an expert JEE tutor with 15 years of experience 
teaching Physics, Chemistry and Mathematics for JEE Main 
and JEE Advanced.

Your teaching style:
- Always start by mentioning Subject and Chapter name
- Break every solution into clear numbered steps
- Explain the CONCEPT behind each step, not just the answer
- Use simple language a Class 11/12 student understands
- Highlight common mistakes students make on this topic
- At the en
d always give 1 similar practice problem
- If not JEE related say: I only help with JEE subjects!
- Support Hinglish if student writes in Hinglish

IMPORTANT - Math Formatting Rules:
- Always wrap ALL math equations in $$ symbols
- Example: $$F = ma$$
- Example: $$v^2 = u^2 + 2as$$
- Example: $$x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$$
- Every formula MUST be inside $$ $$

IMPORTANT - Diagram Rules:
STRICT DIAGRAM RULES - FOLLOW EXACTLY:
- NEVER draw diagrams using text, ASCII art, or symbols like + - | 
- NEVER use boxes made of dashes or pipes
- The app has a built in diagram engine
- ONLY use DIAGRAM_JSON to request diagrams
- Always end with DIAGRAM_JSON if topic has a visual concept

DIAGRAM_JSON format - copy exactly:
DIAGRAM_JSON:
{"type": "sine_wave", "params": {"amplitude": 1, "frequency": 1, "title": "Sine Wave y = A sin(Bx+C)+D"}}

Available types: sine_wave, projectile, graph, force_diagram, electric_field, lens_diagram, circle

Available diagram types and their params:
- "sine_wave": {"amplitude": 1, "frequency": 1, "title": "Wave"}
- "projectile": {"angle": 45, "v0": 20}
- "circle": {"radius": 5, "title": "Circle"}
- "force_diagram": {"forces": [{"name": "Weight", "angle": 270, "magnitude": 10}, {"name": "Normal", "angle": 90, "magnitude": 10}]}
- "graph": {"equation": "x**2", "x_min": -5, "x_max": 5, "title": "y = x²"}
- "electric_field": {"charge": "positive"}
- "lens_diagram": {"type": "convex"}

Only include DIAGRAM_JSON if it genuinely helps explain the concept.
If no diagram is needed, don't include it.
"""

def generate_diagram(diagram_type, params):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    if diagram_type == "sine_wave":
        amplitude = params.get("amplitude", 1)
        frequency = params.get("frequency", 1)
        title = params.get("title", "Wave")
        x = np.linspace(0, 4 * np.pi, 500)
        y = amplitude * np.sin(frequency * x)
        ax.plot(x, y, color='#00d4ff', linewidth=2)
        ax.axhline(0, color='#444', linewidth=0.8)
        ax.set_title(title, color='white', fontsize=14)
        ax.set_xlabel("Time →", color='white')
        ax.set_ylabel("Amplitude →", color='white')
        ax.grid(True, alpha=0.2, color='gray')

    elif diagram_type == "projectile":
        angle = np.radians(params.get("angle", 45))
        v0 = params.get("v0", 20)
        g = 9.8
        t_max = 2 * v0 * np.sin(angle) / g
        t = np.linspace(0, t_max, 300)
        x = v0 * np.cos(angle) * t
        y = v0 * np.sin(angle) * t - 0.5 * g * t**2
        ax.plot(x, y, color='#00d4ff', linewidth=2, label='Trajectory')
        ax.fill_between(x, y, alpha=0.1, color='#00d4ff')
        ax.annotate('Launch', xy=(x[0], y[0]), color='#00ff9d', fontsize=10)
        ax.annotate('Landing', xy=(x[-1], y[-1]), color='#ff6b6b', fontsize=10)
        max_idx = np.argmax(y)
        ax.annotate(f'Max Height\n{y[max_idx]:.1f}m',
            xy=(x[max_idx], y[max_idx]),
            color='yellow', fontsize=9,
            xytext=(x[max_idx]+2, y[max_idx]),
            arrowprops=dict(arrowstyle='->', color='yellow'))
        ax.set_title(f'Projectile Motion (θ={params.get("angle",45)}°, v₀={v0} m/s)', color='white', fontsize=13)
        ax.set_xlabel("Horizontal Distance (m) →", color='white')
        ax.set_ylabel("Vertical Height (m) →", color='white')
        ax.grid(True, alpha=0.2, color='gray')
        ax.legend(facecolor='#1a1a2e', labelcolor='white')

    elif diagram_type == "graph":
        equation = params.get("equation", "x**2")
        x_min = params.get("x_min", -5)
        x_max = params.get("x_max", 5)
        title = params.get("title", "Graph")
        x = np.linspace(x_min, x_max, 500)
        try:
            y = eval(equation, {"x": x, "np": np, "sin": np.sin,
                "cos": np.cos, "tan": np.tan, "exp": np.exp,
                "log": np.log, "sqrt": np.sqrt, "__builtins__": {}})
            ax.plot(x, y, color='#00d4ff', linewidth=2)
        except Exception:
            ax.text(0.5, 0.5, 'Could not plot equation',
                transform=ax.transAxes, color='white', ha='center')
        ax.axhline(0, color='#666', linewidth=0.8)
        ax.axvline(0, color='#666', linewidth=0.8)
        ax.set_title(title, color='white', fontsize=14)
        ax.set_xlabel("x →", color='white')
        ax.set_ylabel("y →", color='white')
        ax.grid(True, alpha=0.2, color='gray')

    elif diagram_type == "force_diagram":
        forces = params.get("forces", [])
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect('equal')
        colors = ['#00d4ff', '#00ff9d', '#ff6b6b', '#ffd700', '#ff69b4']
        for i, force in enumerate(forces):
            angle = np.radians(force.get("angle", 0))
            magnitude = force.get("magnitude", 5)
            name = force.get("name", "F")
            scale = magnitude * 0.5
            dx = scale * np.cos(angle)
            dy = scale * np.sin(angle)
            color = colors[i % len(colors)]
            ax.annotate('', xy=(dx, dy), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
            ax.text(dx * 1.2, dy * 1.2, f'{name}\n({magnitude}N)',
                color=color, fontsize=9, ha='center')
        circle = plt.Circle((0, 0), 1.2, color='#444', zorder=5)
        ax.add_patch(circle)
        ax.text(0, 0, 'Object', color='white', ha='center',
            va='center', fontsize=8, zorder=6)
        ax.set_title('Force Diagram', color='white', fontsize=14)
        ax.grid(True, alpha=0.15, color='gray')
        ax.axhline(0, color='#333', linewidth=0.5)
        ax.axvline(0, color='#333', linewidth=0.5)

    elif diagram_type == "electric_field":
        charge_type = params.get("charge", "positive")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        for angle in angles:
            if charge_type == "positive":
                ax.annotate('', xy=(3.5*np.cos(angle), 3.5*np.sin(angle)),
                    xytext=(0.5*np.cos(angle), 0.5*np.sin(angle)),
                    arrowprops=dict(arrowstyle='->', color='#00d4ff', lw=1.5))
            else:
                ax.annotate('', xy=(0.5*np.cos(angle), 0.5*np.sin(angle)),
                    xytext=(3.5*np.cos(angle), 3.5*np.sin(angle)),
                    arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=1.5))
        symbol = '+' if charge_type == "positive" else '-'
        color = '#00d4ff' if charge_type == "positive" else '#ff6b6b'
        circle = plt.Circle((0, 0), 0.4, color=color, zorder=5)
        ax.add_patch(circle)
        ax.text(0, 0, symbol, color='white', ha='center',
            va='center', fontsize=16, fontweight='bold', zorder=6)
        ax.set_title(f'Electric Field — {charge_type.title()} Charge',
            color='white', fontsize=13)
        ax.axis('off')

    elif diagram_type == "lens_diagram":
        lens_type = params.get("type", "convex")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-5, 5)
        ax.axhline(0, color='#666', linewidth=0.8, linestyle='--', label='Principal Axis')
        ax.axvline(0, color='white', linewidth=2)
        f = 3 if lens_type == "convex" else -3
        ax.axvline(f, color='#ffd700', linewidth=1, linestyle=':', alpha=0.7)
        ax.axvline(-f, color='#ffd700', linewidth=1, linestyle=':', alpha=0.7)
        ax.text(f, -4.5, f'F ({f})', color='#ffd700', ha='center', fontsize=9)
        ax.text(-f, -4.5, f'F\' ({-f})', color='#ffd700', ha='center', fontsize=9)
        if lens_type == "convex":
            theta = np.linspace(-np.pi/3, np.pi/3, 100)
            ax.plot(0.5*np.sin(theta), 3*np.cos(theta) - 3*np.cos(np.pi/3) - 1,
                color='#00ff9d', linewidth=3)
            ax.plot(-0.5*np.sin(theta), 3*np.cos(theta) - 3*np.cos(np.pi/3) - 1,
                color='#00ff9d', linewidth=3)
        ax.set_title(f'{lens_type.title()} Lens Ray Diagram',
            color='white', fontsize=13)
        ax.grid(True, alpha=0.15, color='gray')
        ax.legend(facecolor='#1a1a2e', labelcolor='white')

    elif diagram_type == "circle":
        radius = params.get("radius", 5)
        title = params.get("title", "Circle")
        theta = np.linspace(0, 2*np.pi, 300)
        ax.plot(radius*np.cos(theta), radius*np.sin(theta),
            color='#00d4ff', linewidth=2)
        ax.plot([0, radius], [0, 0], color='#00ff9d',
            linewidth=2, label=f'r = {radius}')
        ax.set_aspect('equal')
        ax.set_title(title, color='white', fontsize=14)
        ax.legend(facecolor='#1a1a2e', labelcolor='white')
        ax.grid(True, alpha=0.2, color='gray')

    plt.tight_layout()
    return fig

def render_message(content):
    diagram_data = None
    if "DIAGRAM_JSON:" in content:
        parts = content.split("DIAGRAM_JSON:")
        content = parts[0].strip()
        if len(parts) > 1:
            try:
                json_str = parts[1].strip()
                json_str = json_str.split('\n\n')[0].strip()
                diagram_data = json.loads(json_str)
            except Exception:
                diagram_data = None

    parts = re.split(r'(\$\$.*?\$\$)', content, flags=re.DOTALL)
    for part in parts:
        if part.startswith('$$') and part.endswith('$$'):
            st.latex(part[2:-2].strip())
        elif part.strip():
            st.markdown(part)

    return diagram_data

st.set_page_config(page_title="JEE AI Tutor", page_icon="🎓", layout="centered")
st.title("🎓 JEE AI Tutor")
st.caption("Your personal AI tutor for JEE Main & Advanced")
st.divider()

subject = st.selectbox("Select Subject:",
    ["All Subjects", "Physics", "Chemistry", "Mathematics"])

uploaded_file = st.file_uploader("📎 Attach file (optional)",
    type=["pdf", "png", "jpg", "jpeg", "txt"])

file_content = ""
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        pdf_bytes = uploaded_file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        for page in pdf_reader.pages:
            file_content += page.extract_text()
        st.success(f"✅ PDF read! ({len(pdf_reader.pages)} pages)")
    elif uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
        image = Image.open(uploaded_file)
        st.image(image, width=400)
        file_content = "[Student uploaded an image of a problem]"
        st.success("✅ Image attached!")
    elif uploaded_file.type == "text/plain":
        file_content = uploaded_file.read().decode("utf-8")
        st.success("✅ File read!")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Namaste! 🙏 Ask me any JEE doubt from Physics, Chemistry or Maths!\n\nI explain step by step with auto-generated diagrams! 📊"
    })

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        diagram_data = render_message(msg["content"])
        if diagram_data and msg["role"] == "assistant":
            try:
                fig = generate_diagram(diagram_data["type"], diagram_data["params"])
                st.pyplot(fig)
                plt.close(fig)
            except Exception:
                pass

if question := st.chat_input("Type your JEE doubt here..."):
    if subject != "All Subjects":
        full_question = f"[{subject}] {question}"
    else:
        full_question = question

    if file_content:
        full_question += "\n\nAttached content:\n" + file_content

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):
            try:
                groq_messages = [{"role": "system", "content": JEE_PROMPT}]
                for msg in st.session_state.messages[:-1]:
                    groq_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                groq_messages.append({"role": "user", "content": full_question})

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=groq_messages,
                    temperature=0.7,
                    max_tokens=2048
                )

                answer = response.choices[0].message.content
                diagram_data = render_message(answer)

                if diagram_data:
                    try:
                        with st.spinner("Drawing diagram... ✏️"):
                            fig = generate_diagram(
                                diagram_data["type"],
                                diagram_data["params"]
                            )
                            st.pyplot(fig)
                            plt.close(fig)
                    except Exception as e:
                        st.info("💡 Diagram could not be generated for this topic")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Something went wrong! Error: {e}")

with st.sidebar:
    st.header("📚 JEE AI Tutor")
    st.write("⚛️ Physics")
    st.write("🧪 Chemistry")
    st.write("📐 Mathematics")
    st.divider()
    st.write("🤖 Llama 3.3 70B")
    st.write("📊 Auto diagrams")
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()