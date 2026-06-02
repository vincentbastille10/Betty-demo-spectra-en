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
    r"(together\s*ai|hiccup|traceback|stack\s*trace|\bexception\b"
    r"|payment\s*required|insufficient\s*(?:credit|fund)|rate\s*limit"
    r"|technical\s*(?:error|issue|problem|difficult)|\bapi\s*(?:error|key)\b"
    r"|server\s*(?:error|issue|down)"
    # Codes d'erreur HTTP uniquement quand ils accompagnent un mot d'erreur,
    # pour ne JAMAIS étouffer une vraie réponse LLM contenant un nombre ($500, 402 clients…).
    r"|(?:error|status|code|http)\s*[:#]?\s*(?:401|402|403|404|429|500|502|503)\b"
    r"|\b(?:401|402|403|404|429|500|502|503)\s*(?:error|payment))",
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


NON_NAMES = {"i", "im", "a", "an", "the", "my", "me", "you", "we", "it", "its", "yes", "yeah",
             "yep", "ok", "okay", "sure", "no", "nope", "maybe", "thanks", "thank", "hi", "hello",
             "hey", "please", "well", "so", "and", "but", "just", "here", "there"}


def is_greeting(t):
    return _norm(t) in GREETINGS


def detect_ask(text):
    """Quel champ la question précédente de Betty visait-elle ?"""
    t = (text or "").lower()
    if "your name" in t or "who am i chatting" in t or "may i ask your name" in t or "first name" in t:
        return "name"
    if "email" in t:
        return "email"
    if "phone" in t:
        return "phone"
    if "business" in t or "industry" in t:
        return "business"
    if "looking to" in t or "what do you need" in t or "what brings you" in t:
        return "need"
    return None


def bare_name(text):
    """Un nom donné seul ('vincent', 'vincent bastille') quand Betty a demandé le nom."""
    if find_email(text) or find_phone(text):
        return ""
    words = re.findall(r"[A-Za-zÀ-ÿ'\-]+", text)
    if not words or len(words) > 2:
        return ""
    for w in words:
        if _norm(w) not in NON_NAMES and _norm(w) not in GREETINGS and len(w) >= 2:
            return w.capitalize()
    return ""


def strip_capture(text):
    return re.sub(r"\s*CAPTURE:.*$", "", str(text or ""), flags=re.IGNORECASE | re.DOTALL).strip()


def rebuild_lead(history, message):
    """Reconstruit le lead à partir de TOUT l'historique (stateless serverless).
    Tient compte de ce que Betty vient de demander pour capter les réponses
    courtes : 'vincent' juste après 'what's your name?' => name=Vincent."""
    lead = {"name": "", "email": "", "phone": "", "business": "", "need": ""}
    seq = [(m.get("role"), m.get("content", "") or "")
           for m in history if m.get("role") in ("user", "assistant")]
    if message:
        seq.append(("user", message))

    last_ask = None  # champ demandé par le dernier message de Betty
    for role, content in seq:
        if role == "assistant":
            cap = CAPTURE_RE.search(content)
            if cap:
                if cap.group(1).strip(): lead["name"]  = cap.group(1).strip()
                if cap.group(2).strip(): lead["email"] = cap.group(2).strip()
                if cap.group(3).strip(): lead["phone"] = cap.group(3).strip()
            last_ask = detect_ask(content)
            continue

        # role == "user" — extraction regex (toujours active)
        if not lead["email"]:    lead["email"]    = find_email(content)
        if not lead["phone"]:    lead["phone"]    = find_phone(content)
        if not lead["name"]:     lead["name"]     = find_name(content)
        if not lead["business"]: lead["business"] = find_business(content)

        # Capture CONTEXTUELLE des réponses courtes à une question précise
        if last_ask and not is_negative(content) and not is_greeting(content):
            if last_ask == "name" and not lead["name"]:
                lead["name"] = bare_name(content)
            elif last_ask == "business" and not lead["business"] and len(content.split()) <= 8:
                lead["business"] = content.strip()[:80]
            elif last_ask == "need" and not lead["need"]:
                lead["need"] = content.strip()[:120]
        last_ask = None  # consommé
    return lead


def _last_bot(history):
    for m in reversed(history):
        if m.get("role") == "assistant":
            return (m.get("content", "") or "").lower()
    return ""


def fallback_reply(lead, history, message):
    """LLM indisponible -> Betty poursuit la qualification, naturellement, en
    anglais, en s'appuyant sur ce qui est DÉJÀ connu (jamais 2x la même question)."""
    name = lead["name"]
    nm = f", {name}" if name else ""
    last_bot = _last_bot(history)
    bot_all = " ".join((m.get("content", "") or "") for m in history if m.get("role") == "assistant").lower()
    neg = is_negative(message)

    if not lead["name"]:
        if neg and "name" in last_bot:
            return "No worries — what's the best email or phone to reach you on, then? 🙂"
        return "Happy to help! 🙂 I'm Betty — what's your name?"
    # On ne demande le secteur qu'une seule fois.
    if not lead["business"] and "business" not in bot_all and "industry" not in bot_all:
        return f"Great to meet you{nm}! 🙂 What kind of business are you in?"
    if not lead["email"] and not lead["phone"]:
        if neg:
            return f"No problem{nm} — whenever you're ready, just drop your email here and our team will follow up. Anything you'd like to know about MyBetty?"
        return f"Love it{nm} — what's the best email or phone so our team can reach out to you?"
    return f"Perfect{nm}, I've got what I need! 🙌 Our team will reach out to you very soon.\n{SIGNUP_LINK}"


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


def _ping_first_provider():
    """Diagnostic : renvoie le statut HTTP du 1er provider (aucune clé exposée)."""
    provs = providers()
    if not provs:
        return {"error": "no_provider_key_set"}
    p = provs[0]
    try:
        r = requests.post(
            p["url"],
            headers={"Authorization": f"Bearer {p['key']}", "Content-Type": "application/json"},
            json={"model": p["model"], "max_tokens": 5, "messages": [{"role": "user", "content": "ping"}]},
            timeout=15,
        )
        return {"provider": p["name"], "model": p["model"], "http": r.status_code, "ok": bool(r.ok)}
    except Exception as e:
        return {"provider": p["name"], "model": p["model"], "error": type(e).__name__}


@app.route("/api/debug")
def debug():
    info = {
        "pack_exists": os.path.exists(YAML_PATH),
        "providers":   [p["name"] for p in providers()],
        "mj_set":      bool(os.environ.get("MJ_APIKEY_PUBLIC") and os.environ.get("MJ_APIKEY_PRIVATE")),
        "model":       os.environ.get("LLM_MODEL", "deepseek-ai/DeepSeek-V3"),
    }
    if request.args.get("ping") == "1":
        info["llm_check"] = _ping_first_provider()
    return jsonify(info)


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
