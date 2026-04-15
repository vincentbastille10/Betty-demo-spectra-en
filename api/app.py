from flask import Flask, render_template, request, jsonify
import requests, yaml, os, re, traceback

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR   = os.path.join(BASE_DIR, "static")
YAML_PATH    = os.path.join(BASE_DIR, "pack", "betty_spectra.yaml")
LEAD_EMAIL   = "spectramediabots@gmail.com"

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

def load_pack():
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def extract_lead(text):
    match = re.search(
        r'CAPTURE:\s*name=\[([^\]]*)\]\s*email=\[([^\]]*)\]\s*phone=\[([^\]]*)\]',
        text, re.IGNORECASE
    )
    if match:
        return {"name": match.group(1).strip(), "email": match.group(2).strip(), "phone": match.group(3).strip()}
    return None

def send_lead_email(lead):
    mj_public  = os.environ.get("MJ_APIKEY_PUBLIC", "")
    mj_private = os.environ.get("MJ_APIKEY_PRIVATE", "")
    if not mj_public or not mj_private:
        return False
    body = (
        f"🎯 New lead captured by Betty (EN Demo)\n\n"
        f"Name : {lead.get('name') or '-'}\n"
        f"Email: {lead.get('email') or '-'}\n"
        f"Phone: {lead.get('phone') or '-'}\n\n"
        f"---\nCaptured via betty-demo-spectra-en.vercel.app"
    )
    try:
        r = requests.post(
            "https://api.mailjet.com/v3.1/send",
            auth=(mj_public, mj_private),
            json={"Messages": [{
                "From": {"Email": LEAD_EMAIL, "Name": "Betty Demo EN"},
                "To":   [{"Email": LEAD_EMAIL}],
                "Subject": f"🎯 Lead Betty EN: {lead.get('name') or 'New contact'}",
                "TextPart": body
            }]},
            timeout=10
        )
        return r.ok
    except Exception:
        return False


@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/healthz")
def healthz():
    return "ok", 200

@app.route("/api/debug")
def debug():
    return jsonify({
        "pack_exists": os.path.exists(YAML_PATH),
        "api_key_set": bool(os.environ.get("TOGETHER_API_KEY")),
        "mj_set":      bool(os.environ.get("MJ_APIKEY_PUBLIC")),
        "model":       os.environ.get("LLM_MODEL", "(not set)"),
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        payload      = request.get_json(silent=True) or {}
        user_message = (payload.get("message") or "").strip()
        history      = payload.get("history", [])

        if not user_message:
            return jsonify({"response": "I'm listening 🙂"})

        try:
            pack          = load_pack()
            system_prompt = pack.get("prompt", "")
        except Exception as e:
            return jsonify({"error": f"Cannot load YAML: {e}"}), 500

        api_key    = os.environ.get("TOGETHER_API_KEY", "")
        model      = os.environ.get("LLM_MODEL", "deepseek-ai/DeepSeek-V3")
        max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "200"))

        if not api_key:
            return jsonify({"error": "TOGETHER_API_KEY missing"}), 500

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-12:]:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message})

        try:
            resp = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "max_tokens": max_tokens, "messages": messages},
                timeout=25
            )
        except requests.exceptions.Timeout:
            return jsonify({"error": "Together AI timeout (>25s)"}), 504
        except Exception as e:
            return jsonify({"error": f"Network error: {e}"}), 500

        if not resp.ok:
            return jsonify({"error": f"Together AI error {resp.status_code}", "detail": resp.text}), 502

        try:
            raw_reply = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return jsonify({"error": f"Parse error: {e}", "raw": resp.text[:300]}), 500

        lead        = extract_lead(raw_reply)
        clean_reply = re.sub(r'\nCAPTURE:.*', '', raw_reply, flags=re.IGNORECASE | re.DOTALL).strip()
        lead_captured = False
        if lead and (lead.get("email") or lead.get("phone")):
            lead_captured = send_lead_email(lead)

        return jsonify({"response": clean_reply, "lead_captured": lead_captured})

    except Exception as e:
        return jsonify({"error": "Unexpected error", "detail": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
