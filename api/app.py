from flask import Flask, render_template, request, jsonify
import requests, yaml, os, re

# ─── Paths ──────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR   = os.path.join(BASE_DIR, "static")
YAML_PATH    = os.path.join(BASE_DIR, "pack", "betty_spectra.yaml")
LEAD_EMAIL   = os.environ.get("LEAD_EMAIL", "spectramediabots@gmail.com")
SIGNUP_LINK  = os.environ.get("SIGNUP_LINK", "https://MyBetty.online")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

DEFAULT_PROMPT = (
    "You are Betty, the warm, human sales assistant for MyBetty.online. "
    "Reply in ENGLISH, 1-3 short sentences, never robotic. Engage the visitor, "
    "show interest in their business, and naturally collect their first name and "
    "email or phone so the team can follow up. Always end with one short question. "
    "Never mention anything technical."
)


def load_prompt():
    try:
        with open(YAML_PATH, "r", encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("prompt", "").strip() or DEFAULT_PROMPT
    except Exception:
        return DEFAULT_PROMPT


# ─── LLM providers (Together -> Groq -> OpenAI-compatible) ──────────
def providers():
    provs = []
    tog = os.environ.get("TOGETHER_API_KEY", "").strip()
    if tog:
        provs.append({"name": "together", "url": "https://api.together.xyz/v1/chat/completions",
                      "key": tog, "model": os.environ.get("LLM_MODEL", "deepseek-ai/DeepSeek-V3")})
    grq = os.environ.get("GROQ_API_KEY", "").strip()
    if grq:
        provs.append({"name": "groq", "url": "https://api.groq.com/openai/v1/chat/completions",
                      "key": grq, "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")})
    oai = os.environ.get("OPENAI_API_KEY", "").strip()
    if oai:
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        provs.append({"name": "openai", "url": base + "/chat/completions",
                      "key": oai, "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini")})
    return provs


# ─── Anti-leak guard: no technical term ever reaches the visitor ────
_LEAK_RE = re.compile(
    r"(together\s*ai|hiccup|traceback|stack\s*trace|\bexception\b|status\s*code|"
    r"payment\s*required|insufficient\s*(?:credit|fund)|"
    r"technical\s*(?:error|issue|problem|difficult)|api\s*(?:error|key)|"
    r"server\s*(?:error|issue|down)|rate\s*limit|\b(?:401|402|403|429|500|502|503)\b)",
    re.IGNORECASE,
)
SAFE_FALLBACK = "I'd love to help you with that! 🙂 Could you share your name and the best email to reach you?"


def looks_like_error(text):
    return bool(_LEAK_RE.search(str(text or "")))


def ensure_clean(text):
    t = str(text or "").strip()
    return SAFE_FALLBACK if (not t or looks_like_error(t)) else t


# ─── Lead extraction (typo-tolerant, stateless) ────────────────────
EMAIL_RE   = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE   = re.compile(r"(?<!\d)(\+?\d[\d\s().\-]{6,18}\d)(?!\d)")
NAME_RE    = re.compile(r"\b(?:i\s*am|i'?m|my\s+name\s+is|name'?s|this\s+is|call\s+me|it'?s)\s+([A-Za-z][A-Za-z'\-]{1,24})", re.IGNORECASE)
CAPTURE_RE = re.compile(r"CAPTURE:\s*name=\[([^\]]*)\]\s*email=\[([^\]]*)\]\s*phone=\[([^\]]*)\]", re.IGNORECASE)
NEGATIVES  = {"no", "nope", "nah", "skip", "later", "none", "na", "n/a", "pass", "not now", "rather not", "prefer not", "no thanks"}
GREETINGS  = {"hi", "hello", "hey", "yo", "hiya", "howdy", "sup"}
BUSINESS_HINTS = {"shop", "store", "restaurant", "agency", "salon", "clinic", "dentist", "lawyer", "real estate",
                  "coach", "ecommerce", "saas", "startup", "consultant", "freelance", "gym", "hotel", "garage",
                  "plumber", "electrician", "painter", "construction", "bakery", "cafe", "spa", "studio",
                  "photographer", "accountant", "insurance", "contractor", "hairdresser"}


def _norm(t):
    return re.sub(r"\s+", " ", str(t or "").lower().strip()).rstrip("!.,?")


def is_negative(t):
    return _norm(t) in NEGATIVES


def find_email(t):
    m = EMAIL_RE.search(str(t or ""))
    return m.group(0).strip() if m else ""


def find_phone(t):
    for m in PHONE_RE.finditer(str(t or "")):
        if 7 <= len(re.sub(r"\D", "", m.group(1))) <= 15:
            return m.group(1).strip()
    return ""


def find_name(t):
    m = NAME_RE.search(str(t or ""))
    if m and _norm(m.group(1)) not in GREETINGS:
        return m.group(1).capitalize()
    return ""


def find_business(t):
    n = " " + _norm(t) + " "
    if len(n.split()) <= 6:
        for h in BUSINESS_HINTS:
            if " " + h + " " in n:
                return str(t).strip()[:80]
    return ""


def strip_capture(text):
    return re.sub(r"\s*CAPTURE:.*$", "", str(text or ""), flags=re.IGNORECASE | re.DOTALL).strip()


def rebuild_lead(history, message):
    """Reconstruit le lead à partir de tout l'historique (stateless serverless)."""
    lead = {"name": "", "email": "", "phone": "", "business": "", "need": ""}
    # 1) marqueurs CAPTURE émis par Betty (signal le plus fiable)
    for m in history:
        if m.get("role") == "assistant":
            cap = CAPTURE_RE.search(m.get("content", "") or "")
            if cap:
                if cap.group(1).strip(): lead["name"] = cap.group(1).strip()
                if cap.group(2).strip(): lead["email"] = cap.group(2).strip()
                if cap.group(3).strip(): lead["phone"] = cap.group(3).strip()
    # 2) messages du visiteur (regex tolérant)
    user_texts = [m.get("content", "") for m in history if m.get("role") == "user"]
    if message:
        user_texts.append(message)
    for t in user_texts:
        lead["email"]    = lead["email"]    or find_email(t)
        lead["phone"]    = lead["phone"]    or find_phone(t)
        lead["name"]     = lead["name"]     or find_name(t)
        lead["business"] = lead["business"] or find_business(t)
    return lead


def _last_bot(history):
    for m in reversed(history):
        if m.get("role") == "assistant":
            return (m.get("content", "") or "").lower()
    return ""


def fallback_reply(lead, history, message):
    """LLM indisponible -> on poursuit la capture, proprement, en anglais."""
    name = lead["name"]
    nm = f", {name}" if name else ""
    last_bot = _last_bot(history)
    neg = is_negative(message)

    if not lead["name"]:
        if neg and "name" in last_bot:
            return "No worries! 🙂 What's the best email or phone to reach you, then?"
        return "I'd love to help with that! 🙂 First, who am I chatting with — what's your name?"
    if not lead["email"] and not lead["phone"]:
        if neg and ("email" in last_bot or "phone" in last_bot or "reach you" in last_bot):
            return f"All good{nm} — if you'd like, just drop your email anytime and our team will follow up. What would you like to know about MyBetty?"
        return f"Great to meet you{nm}! What's the best email or phone so we can send you the details?"
    return f"Perfect{nm}, I've noted your details — our team will reach out to you very soon. 🙌\n{SIGNUP_LINK}"


# ─── Lead delivery (Mailjet) ───────────────────────────────────────
def send_lead_email(lead):
    mj_public  = os.environ.get("MJ_APIKEY_PUBLIC", "")
    mj_private = os.environ.get("MJ_APIKEY_PRIVATE", "")
    if not mj_public or not mj_private:
        return False
    body = (
        f"🎯 New lead captured by Betty (EN Demo)\n\n"
        f"Name    : {lead.get('name') or '-'}\n"
        f"Email   : {lead.get('email') or '-'}\n"
        f"Phone   : {lead.get('phone') or '-'}\n"
        f"Business: {lead.get('business') or '-'}\n\n"
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
        return bool(r.ok)
    except Exception:
        return False


# ─── LLM call (multi-provider, never returns a leak) ───────────────
def call_llm(system_prompt, history, message):
    msgs = [{"role": "system", "content": system_prompt}]
    for m in history[-12:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": message})

    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "220"))
    for p in providers():
        try:
            r = requests.post(
                p["url"],
                headers={"Authorization": f"Bearer {p['key']}", "Content-Type": "application/json"},
                json={"model": p["model"], "max_tokens": max_tokens, "temperature": 0.6, "messages": msgs},
                timeout=25,
            )
            if not r.ok:
                app.logger.warning("provider %s -> HTTP %s", p["name"], r.status_code)
                continue
            raw = (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
            if raw and not looks_like_error(raw):
                return raw
        except Exception as e:
            app.logger.warning("provider %s -> %s", p["name"], type(e).__name__)
            continue
    return None  # aucun LLM -> bascule sur le fallback déterministe


# ─── Routes ────────────────────────────────────────────────────────
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
        "providers":   [p["name"] for p in providers()],
        "mj_set":      bool(os.environ.get("MJ_APIKEY_PUBLIC") and os.environ.get("MJ_APIKEY_PRIVATE")),
        "model":       os.environ.get("LLM_MODEL", "deepseek-ai/DeepSeek-V3"),
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    # Tout est protégé : le visiteur ne voit JAMAIS d'erreur technique,
    # et la capture du lead prime sur la génération IA.
    try:
        payload = request.get_json(silent=True) or {}
        message = (payload.get("message") or "").strip()
        history = payload.get("history") or []
        if not isinstance(history, list):
            history = []

        if not message:
            return jsonify({"response": "Hi! 🙂 What kind of business are you in?", "lead_captured": False})

        lead_before = rebuild_lead(history, "")       # état avant ce message
        lead        = rebuild_lead(history, message)  # avec le message courant

        raw = call_llm(load_prompt(), history, message)
        if raw:
            cap = CAPTURE_RE.search(raw)
            if cap:
                if cap.group(1).strip(): lead["name"]  = cap.group(1).strip()
                if cap.group(2).strip(): lead["email"] = cap.group(2).strip()
                if cap.group(3).strip(): lead["phone"] = cap.group(3).strip()
            reply = strip_capture(raw)
        else:
            reply = fallback_reply(lead, history, message)

        reply = ensure_clean(reply)

        # Envoi du lead une seule fois : la 1re fois qu'un email/téléphone apparaît.
        new_contact = bool((lead["email"] and not lead_before["email"]) or
                           (lead["phone"] and not lead_before["phone"]))
        lead_captured = False
        if new_contact and (lead["email"] or lead["phone"]):
            lead_captured = send_lead_email(lead)

        return jsonify({"response": reply, "lead_captured": lead_captured})

    except Exception as e:
        # Dernier rempart : message propre, capture du lead, zéro fuite.
        app.logger.warning("chat fatal: %r", e)
        return jsonify({"response": SAFE_FALLBACK, "lead_captured": False})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
