from flask import Flask, render_template, request, jsonify
import requests, yaml, os, re, time

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

REAL_ESTATE_PROMPT = """
You are Betty, a warm, natural virtual receptionist for a US real estate agent.
This is a live product demonstration: the visitor should experience how Betty qualifies a real buyer, seller, investor, or valuation request.

Your goal is to understand the real estate project before asking for contact details.
Ask exactly one short question at a time and reply in natural American English.

FLOW:
1. Explain the demo context, then ask whether the visitor wants to buy, sell, invest, or request a home valuation.
2. Ask for the city or neighborhood.
3. Ask for the property type.
4. Ask for the budget, price range, or estimated property value, depending on intent.
5. Ask when they plan to move, invest, or sell.
6. Only after those useful details, ask for their first name.
7. Ask for one preferred contact method: email OR mobile number.
8. Confirm with a concise qualified lead summary and explain that this is what the agent receives instantly.
9. Finish with https://MyBetty.online/real-estate

RULES:
- Never ask what kind of business they are in.
- Never ask for contact details before collecting useful real estate context.
- Never request both email and phone; one is enough.
- Answer a visitor's direct question before continuing qualification.
- Never invent listings, prices, availability, or market facts.
- Never say you are an AI.
- Use 1 to 3 short sentences per turn.
- Avoid artificial phrases such as 'Love it' after receiving a name.
- MyBetty costs $149/month, includes a 7-day free trial, requires a card, and does not charge during the trial.
- Setup takes approximately 3 minutes.
- When contact information is captured, add as the final hidden line:
  CAPTURE: name=[firstname] email=[email or empty] phone=[phone or empty]
""".strip()


def load_config():
    try:
        with open(YAML_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_knowledge_base():
    return load_config().get("knowledge_base", {}) or {}


def _knowledge_for_prompt(mode):
    lines = []
    for entry in load_knowledge_base().get("entries", []):
        modes = entry.get("modes") or ["generic", "real-estate"]
        if mode in modes and entry.get("answer"):
            lines.append(f"- {entry.get('id', 'fact')}: {entry['answer']}")
    return "\n".join(lines)


def load_prompt(mode="generic"):
    config = load_config()
    base = REAL_ESTATE_PROMPT if mode == "real-estate" else (
        str(config.get("prompt") or "").strip() or DEFAULT_PROMPT
    )
    facts = _knowledge_for_prompt(mode)
    if not facts:
        return base
    return (
        base
        + "\n\nVERIFIED KNOWLEDGE BASE — source of truth; never contradict or extend it:\n"
        + facts
    )


def find_kb_answer(message, mode="generic"):
    text = _norm(message)
    if not text:
        return ""
    best = None
    best_score = -1
    base = load_knowledge_base()
    for entry in base.get("entries", []):
        modes = entry.get("modes") or ["generic", "real-estate"]
        if mode not in modes:
            continue
        matched = [
            trigger for trigger in entry.get("triggers", [])
            if _norm(trigger) and _norm(trigger) in text
        ]
        if not matched:
            continue
        score = int(entry.get("priority", 0)) * 1000 + max(len(_norm(t)) for t in matched)
        if score > best_score:
            best = entry
            best_score = score
    if best:
        return str(best.get("answer") or "").strip()

    question_starts = (
        "what ", "how ", "does ", "can ", "is ", "are ", "do ",
        "tell me ", "i want to know ", "could you "
    )
    product_terms = ("betty", "mybetty", "chatbot", "assistant", "subscription", "service")
    if (
        any(term in text for term in product_terms)
        and ("?" in str(message) or text.startswith(question_starts))
    ):
        return str(base.get("unknown_answer") or "").strip()
    return ""


def find_qualification_profile(activity):
    text = _norm(activity)
    if not text:
        return {}
    for profile in load_config().get("qualification_profiles", []):
        if any(_norm(trigger) in text for trigger in profile.get("triggers", [])):
            return profile
    return {}


def combine_knowledge_and_flow(answer, flow_reply):
    answer = str(answer or "").strip()
    flow_reply = str(flow_reply or "").strip()
    return f"{answer}\n\n{flow_reply}" if answer else flow_reply


# ─── LLM providers (Together -> Groq -> OpenAI-compatible) ──────────
# Modèles Together connus comme valides : si LLM_MODEL est mal configuré
# (slug déprécié -> HTTP 400), Betty bascule automatiquement sur ceux-ci,
# pour qu'une vraie réponse LLM reste possible sans toucher aux env vars.
TOGETHER_FALLBACK_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "meta-llama/Llama-3.1-8B-Instruct-Turbo",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
]


def providers():
    """Together est prioritaire dès qu'une clé existe.
    Sans clé, ou si Together échoue, le parcours déterministe prend le relais."""
    key = os.environ.get("TOGETHER_API_KEY", "").strip()
    if not key:
        return []
    model = os.environ.get(
        "LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    ).strip()
    return [{
        "name": "together",
        "url": "https://api.together.xyz/v1/chat/completions",
        "key": key,
        "model": model,
    }]


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
SAFE_FALLBACK = "Hi! 🙂 To make this demo useful, what kind of business are you in?"


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
    if (
        "most useful detail" in t
        or "which detail" in t
        or ("budget" in t and "timeline" in t)
        or "qualify first" in t
    ):
        return "qualifier"
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
    lead = {"name": "", "email": "", "phone": "", "business": "", "need": "", "qualifier": ""}
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
        knowledge_turn = bool(
            find_kb_answer(content, "generic") or
            find_kb_answer(content, "real-estate")
        )
        if last_ask and not is_negative(content) and not is_greeting(content) and not knowledge_turn:
            if last_ask == "name" and not lead["name"]:
                lead["name"] = bare_name(content)
            elif last_ask == "business" and not lead["business"] and len(content.split()) <= 8:
                lead["business"] = content.strip()[:80]
            elif last_ask == "need" and not lead["need"]:
                lead["need"] = content.strip()[:120]
            elif last_ask == "qualifier" and not lead["qualifier"]:
                lead["qualifier"] = content.strip()[:120]
        last_ask = None  # consommé
    return lead


def _last_bot(history):
    for m in reversed(history):
        if m.get("role") == "assistant":
            return (m.get("content", "") or "").lower()
    return ""


def detect_real_estate_ask(text):
    t = (text or "").lower()
    if ("buy" in t and "sell" in t) or "home valuation" in t:
        return "intent"
    if "city or neighborhood" in t or "area is the property" in t:
        return "location"
    if "property type" in t or "kind of property" in t:
        return "property_type"
    if "price range" in t or "estimated property value" in t or "rough value" in t:
        return "budget"
    if "when would" in t or "timeline" in t or "how soon" in t:
        return "timeline"
    return None


def normalize_real_estate_intent(value):
    n = _norm(value)
    if "valu" in n or "worth" in n:
        return "valuation"
    if "sell" in n or "list" in n:
        return "seller"
    if "invest" in n:
        return "investor"
    if "rent" in n:
        return "renter"
    if "buy" in n or "purchas" in n:
        return "buyer"
    return value.strip()[:40] if value else ""


def rebuild_real_estate_state(history, message):
    state = {"intent": "", "location": "", "property_type": "", "budget": "", "timeline": ""}
    seq = [(m.get("role"), m.get("content", "") or "")
           for m in history if m.get("role") in ("user", "assistant")]
    if message:
        seq.append(("user", message))
    last_ask = None
    for role, content in seq:
        if role == "assistant":
            last_ask = detect_real_estate_ask(content)
            continue
        knowledge_turn = bool(
            find_kb_answer(content, "generic") or
            find_kb_answer(content, "real-estate")
        )
        if last_ask and not is_greeting(content) and not is_negative(content) and not knowledge_turn:
            value = content.strip()[:120]
            if last_ask == "intent":
                value = normalize_real_estate_intent(value)
            if value:
                state[last_ask] = value
        last_ask = None
    return state


def fallback_real_estate_reply(lead, state):
    name = lead.get("name", "")
    if not state["intent"]:
        return ("Hi! I'm Betty, a virtual receptionist for real estate agents. "
                "For this demo, are you looking to buy, sell, invest, or request a home valuation?")
    if not state["location"]:
        return "Great — which city or neighborhood is your real estate project in?"
    if not state["property_type"]:
        return "What property type are you interested in — house, condo, apartment, land, or something else?"
    if not state["budget"]:
        if state["intent"] in ("seller", "valuation"):
            return "Do you have a rough estimate of the property's current value?"
        return "What price range are you considering?"
    if not state["timeline"]:
        return "When would you ideally like to move forward?"
    if not name:
        return "Perfect — I now have useful context for the agent. What's your first name?"
    if not lead.get("email") and not lead.get("phone"):
        return f"Thanks, {name}. What's the best email or mobile number for a local agent to reach you?"
    label = state["intent"].replace("_", " ").title()
    contact = lead.get("email") or lead.get("phone")
    return (
        f"Perfect, {name}! Here's the qualified lead summary the agent would receive instantly:\n"
        f"• {label} · {state['location']} · {state['property_type']}\n"
        f"• Budget/value: {state['budget']} · Timeline: {state['timeline']}\n"
        f"• Contact: {contact}\n\n"
        "That's how Betty turns an anonymous visitor into a ready-to-call lead — 24/7.\n"
        "https://MyBetty.online/real-estate"
    )


def fallback_reply(lead, history, message):
    """LLM indisponible -> Betty poursuit la qualification, naturellement, en
    anglais, en s'appuyant sur ce qui est DÉJÀ connu (jamais 2x la même question)."""
    name = lead["name"]
    nm = f", {name}" if name else ""
    last_bot = _last_bot(history)
    bot_all = " ".join((m.get("content", "") or "") for m in history if m.get("role") == "assistant").lower()
    neg = is_negative(message)

    if not lead["business"]:
        return "Hi! 🙂 To make this demo useful, what kind of business are you in?"
    if not lead["need"]:
        return ("Great — what do you need Betty to qualify or capture on your website: "
                "quote requests, appointments, sales inquiries, registrations, or something else?")
    if not lead["qualifier"]:
        profile = find_qualification_profile(lead["business"])
        if profile.get("question"):
            return profile["question"]
        return ("What's the most useful detail to qualify first — service needed, location, "
                "budget, timeline, or urgency?")
    if not lead["name"]:
        profile = find_qualification_profile(lead["business"])
        value = str(profile.get("value") or "").strip()
        prefix = f"{value} " if value else "Perfect — I now have useful context for your team. "
        return prefix + "What's your first name?"
    if not lead["email"] and not lead["phone"]:
        if neg:
            return f"No problem{nm}. Your demo summary is ready whenever you'd like to continue."
        return f"Thanks{nm}. What's the best email or mobile number for our team to send your activation details?"
    contact = lead["email"] or lead["phone"]
    return (
        f"Perfect{nm}! Here's the qualified summary our team receives:\n"
        f"• Business: {lead['business']}\n"
        f"• Goal: {lead['need']}\n"
        f"• Qualification: {lead['qualifier']}\n"
        f"• Contact: {contact}\n\n"
        f"That's exactly what your own visitors' conversations can produce.\n{SIGNUP_LINK}"
    )


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
        f"Business: {lead.get('business') or '-'}\n"
        f"Goal    : {lead.get('need') or '-'}\n"
        f"Qualify : {lead.get('qualifier') or '-'}\n"
        f"Intent  : {lead.get('intent') or '-'}\n"
        f"Location: {lead.get('location') or '-'}\n"
        f"Property: {lead.get('property_type') or '-'}\n"
        f"Budget  : {lead.get('budget') or '-'}\n"
        f"Timeline: {lead.get('timeline') or '-'}\n\n"
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
                timeout=15,
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


# ─── Garde-fou credits ─────────────────────────────────────────────
# Cette demo est publique et sans authentification : chaque message coute un
# appel LLM facture. Sans limite, un script qui boucle dessus vide le compte.
# Au-dela du quota, on ne renvoie PAS d'erreur : on sert fallback_reply(), le
# repli deterministe, qui est gratuit et capture toujours le lead. Un pirate
# ne coute plus rien ; un vrai prospect a deja eu ses reponses IA.
_HITS: dict = {}
_MAX_LLM_PAR_IP = int(os.environ.get("DEMO_MAX_LLM_PAR_IP", "3"))
_FENETRE_S = int(os.environ.get("DEMO_FENETRE_SECONDES", "600"))


def _client_ip():
    return ((request.headers.get("X-Forwarded-For", "").split(",")[0].strip())
            or request.remote_addr or "?")


def _llm_autorise(ip):
    maintenant = time.time()
    seuil = maintenant - _FENETRE_S
    if len(_HITS) > 5000:      # borne memoire
        _HITS.clear()
    hits = [t for t in _HITS.get(ip, []) if t > seuil]
    if len(hits) >= _MAX_LLM_PAR_IP:
        _HITS[ip] = hits
        return False
    hits.append(maintenant)
    _HITS[ip] = hits
    return True


# ─── Routes ────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/healthz")
def healthz():
    return "ok", 200


def _ping_one(p, model=None):
    mdl = model or p["model"]
    try:
        r = requests.post(
            p["url"],
            headers={"Authorization": f"Bearer {p['key']}", "Content-Type": "application/json"},
            json={"model": mdl, "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]},
            timeout=15,
        )
        return {"provider": p["name"], "model": mdl, "http": r.status_code, "ok": bool(r.ok)}
    except Exception as e:
        return {"provider": p["name"], "model": mdl, "error": type(e).__name__}


def _ping_providers(model=None):
    """Diagnostic : statut HTTP de chaque modèle candidat (aucune clé exposée)."""
    provs = providers()
    if not provs:
        return [{"error": "no_provider_key_set"}]
    if model:  # sonde un slug précis sur le 1er provider
        return [_ping_one(provs[0], model)]
    return [_ping_one(p) for p in provs]


@app.route("/api/debug")
def debug():
    info = {
        "pack_exists": os.path.exists(YAML_PATH),
        "providers":   [p["name"] for p in providers()],
        "mj_set":      bool(os.environ.get("MJ_APIKEY_PUBLIC") and os.environ.get("MJ_APIKEY_PRIVATE")),
        "model_env":   os.environ.get("LLM_MODEL", "(unset)"),
    }
    if request.args.get("ping") == "1":
        info["llm_check"] = _ping_providers(request.args.get("model"))
    return jsonify(info)


@app.route("/api/chat", methods=["POST"])
def chat():
    # Tout est protégé : le visiteur ne voit JAMAIS d'erreur technique,
    # et la capture du lead prime sur la génération IA.
    try:
        payload = request.get_json(silent=True) or {}
        message = (payload.get("message") or "").strip()
        history = payload.get("history") or []
        mode = (payload.get("mode") or "generic").strip().lower()
        if mode not in ("generic", "real-estate"):
            mode = "generic"
        if not isinstance(history, list):
            history = []

        if not message:
            opening = ("Hi! I'm Betty, a virtual receptionist for real estate agents. "
                       "For this demo, are you looking to buy, sell, invest, or request a home valuation?") \
                      if mode == "real-estate" else "Hi! 🙂 To make this demo useful, what kind of business are you in?"
            return jsonify({"response": opening, "lead_captured": False})

        knowledge_answer = find_kb_answer(message, mode)
        qualification_message = "" if knowledge_answer else message

        lead_before = rebuild_lead(history, "")
        lead = rebuild_lead(history, qualification_message)
        real_state = (
            rebuild_real_estate_state(history, qualification_message)
            if mode == "real-estate" else {}
        )
        if real_state:
            lead.update(real_state)

        ip = _client_ip()
        if _llm_autorise(ip):
            raw = call_llm(load_prompt(mode), history, message)
        else:
            raw = None   # quota epuise -> repli deterministe, gratuit
            app.logger.warning("quota LLM depasse pour %s -> fallback", ip)
        if raw:
            cap = CAPTURE_RE.search(raw)
            if cap:
                if cap.group(1).strip(): lead["name"]  = cap.group(1).strip()
                if cap.group(2).strip(): lead["email"] = cap.group(2).strip()
                if cap.group(3).strip(): lead["phone"] = cap.group(3).strip()
            reply = strip_capture(raw)
        else:
            flow_reply = (
                fallback_real_estate_reply(lead, real_state)
                if mode == "real-estate"
                else fallback_reply(lead, history, qualification_message)
            )
            reply = combine_knowledge_and_flow(knowledge_answer, flow_reply)

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
        safe_reply = (
            "Hi! I'm Betty, a virtual receptionist for real estate agents. "
            "Are you looking to buy, sell, invest, or request a home valuation?"
            if locals().get("mode") == "real-estate"
            else SAFE_FALLBACK
        )
        return jsonify({"response": safe_reply, "lead_captured": False})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
