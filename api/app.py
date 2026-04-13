from flask import Flask, render_template, request, jsonify
import requests
import yaml
import os

# ─── Chemins absolus ───────────────────────────────────────────────
# api/app.py est dans /api/ — templates/ et static/ sont à la RACINE
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR   = os.path.join(BASE_DIR, "static")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR,
)

# ─── Config LLM ────────────────────────────────────────────────────
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "").strip()
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"   # ✅ slug corrigé

# ─── Chargement YAML ───────────────────────────────────────────────
YAML_PATH = os.path.join(BASE_DIR, "pack", "betty_spectra.yaml")
with open(YAML_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
SYSTEM_PROMPT = config["prompt"]

# ─── Historique en mémoire (stateless sur Vercel) ──────────────────
CONV_HISTORY: list[dict] = []

# ─── Routes ────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    # ✅ Renommé en `payload` pour éviter la collision avec `result`
    payload      = request.get_json(silent=True) or {}
    user_message = (payload.get("message") or "").strip()

    if not user_message:
        return jsonify({"response": "Je vous écoute 🙂"})

    # ✅ Debug clé API (à retirer en production)
    print(f"[DEBUG] TOGETHER_API_KEY: {TOGETHER_API_KEY[:8]}... ({len(TOGETHER_API_KEY)} chars)")

    if not TOGETHER_API_KEY:
        return jsonify({"response": "Erreur serveur : TOGETHER_API_KEY manquant."}), 500

    # Garde les 8 derniers échanges
    CONV_HISTORY.append({"role": "user", "content": user_message})
    history  = CONV_HISTORY[-8:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    reply = "Petit souci technique… pouvez-vous reformuler ? 🙂"  # valeur par défaut

    try:
        response = requests.post(
            TOGETHER_API_URL,
            headers={
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       LLM_MODEL,
                "messages":    messages,
                "temperature": 0.7,
                "max_tokens":  400,
            },
            timeout=30,
        )

        # ✅ Log du statut HTTP pour debug
        print(f"[DEBUG] Together status: {response.status_code}")

        response.raise_for_status()  # lève HTTPError si 4xx/5xx
        result = response.json()     # ✅ renommé `result` (pas `data`)

        # ✅ try/except imbriqués DANS le try principal, bien ordonnés
        try:
            reply = result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            try:
                reply = result["choices"][0]["text"].strip()
            except (KeyError, IndexError):
                print("[WARN] Format de réponse inattendu :", result)
                reply = "Je rencontre un souci temporaire 🙂"

    except requests.exceptions.HTTPError as e:
        # ✅ Log détaillé pour identifier 401 / 422 / 429 etc.
        print(f"[ERREUR HTTP] {e} | Body: {response.text[:300]}")
        reply = f"Erreur API ({response.status_code}) — vérifiez les logs serveur."

    except requests.exceptions.Timeout:
        print("[ERREUR] Timeout Together AI")
        reply = "L'IA met trop de temps à répondre. Réessayez 🙂"

    except requests.exceptions.ConnectionError:
        print("[ERREUR] Connexion Together AI impossible")
        reply = "Impossible de joindre l'IA. Vérifiez la connexion réseau."

    except Exception as e:
        print(f"[ERREUR inattendue] {type(e).__name__}: {e}")
        reply = "Petit souci technique… pouvez-vous reformuler ? 🙂"

    CONV_HISTORY.append({"role": "assistant", "content": reply})
    return jsonify({"response": reply})


@app.route("/healthz")
def healthz():
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
