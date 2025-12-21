# --- START OF FILE app.py ---
from flask import Flask, render_template, request, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import logging
import json
from datetime import datetime
from dotenv import load_dotenv
import requests # WhatsApp'a yanıt göndermek için eklendi
from langdetect import detect, LangDetectException # Dil tespiti için eklendi

load_dotenv()

# --- MemoryManager'ı file_load.py'dan import et ---
from file_load import MemoryManager

# Fonksiyon importları
# YENİ: get_rag_files_by_lang artık openai_func.py'den geliyor.
from openai_func import (
    openai_func as openai_main_func,
    load_and_process_document_openai,
    clear_rag_context_openai,
    get_processed_filename_openai
)



app = Flask(__name__)
app.secret_key = "supersecret_rag_pro_edition"
app.config['UPLOAD_FOLDER'] = 'uploads_temp'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'xlsx', 'xls', 'docx'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

logging.basicConfig(level=logging.INFO)
logger = app.logger

# MemoryManager örneği
memory_manager = MemoryManager()

# AIRTABLE DEBUG TEST
print("=== AIRTABLE DEBUG ===")
try:
    print(f"Airtable Enabled: {memory_manager.airtable.enabled}")
    print(f"API Key mevcut: {bool(memory_manager.airtable.api_key)}")
    print(f"Base ID: {memory_manager.airtable.base_id}")
    print(f"Table Name: {memory_manager.airtable.table_name}")
    print(f"Table Object: {type(memory_manager.airtable.table)}")
    if memory_manager.airtable.enabled:
        print("✅ Airtable yapılandırması BAŞARILI!")
    else:
        print("⚠️  Airtable KAPALI veya HATALI!")
except Exception as e:
    print(f"❌ Airtable debug hatası: {e}")
print("========================")

# Model tanımı
MODEL_GPT = "GPT-4o-mini"

AVAILABLE_MODELS = {
    MODEL_GPT: {
        "func": openai_main_func,
        "display_name": "GPT-4o-mini",
        "history_name": "GPT",
        "rag_loader": load_and_process_document_openai,
        "rag_clearer": clear_rag_context_openai,
        "rag_filename_getter": get_processed_filename_openai
    }
}

DEFAULT_MODEL = MODEL_GPT

# --- ÇOK DİLLİ RAG YÜKLEME ---
DEV_RAG_FILES_TR = [
    os.path.join(os.path.dirname(__file__), "uploads", "ürünlertürkçe.xlsx"),
    os.path.join(os.path.dirname(__file__), "uploads", "özgenplastik.pdf"),
    os.path.join(os.path.dirname(__file__), "uploads", "sorucevapözgen.pdf"),
]
DEV_RAG_FILES_EN = [
    os.path.join(os.path.dirname(__file__), "uploads", "ürünleringilizce.xlsx"),
    os.path.join(os.path.dirname(__file__), "uploads", "özgenplastiking.pdf"),
    os.path.join(os.path.dirname(__file__), "uploads", "faqingilizce.pdf"),
]

def preload_rag_documents():
    """Uygulama başlarken Türkçe ve İngilizce RAG belgelerini yükler."""
    rag_loader = AVAILABLE_MODELS[DEFAULT_MODEL].get("rag_loader")
    if not rag_loader:
        logger.error("RAG Yükleyici fonksiyonu bulunamadı!")
        return

    # Türkçe belgeler
    logger.info("Türkçe RAG belgeleri yükleniyor...")
    for fpath in DEV_RAG_FILES_TR:
        if os.path.exists(fpath):
            fname = os.path.basename(fpath)
            success, msg = rag_loader(fpath, fname, lang='tr')
            if success:
                logger.info(f"Başarılı (TR): {msg}")
            else:
                logger.error(f"Hata (TR): {msg}")
        else:
            logger.warning(f"Dosya bulunamadı (TR): {fpath}")

    # İngilizce belgeler
    logger.info("İngilizce RAG belgeleri yükleniyor...")
    for fpath in DEV_RAG_FILES_EN:
        if os.path.exists(fpath):
            fname = os.path.basename(fpath)
            success, msg = rag_loader(fpath, fname, lang='en')
            if success:
                logger.info(f"Başarılı (EN): {msg}")
            else:
                logger.error(f"Hata (EN): {msg}")
        else:
            logger.warning(f"Dosya bulunamadı (EN): {fpath}")

preload_rag_documents()
# --- / ÇOK DİLLİ RAG YÜKLEME ---


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    session.setdefault("chat_open", False)
    session.setdefault("chat_history", [])
    session["selected_model"] = DEFAULT_MODEL

    current_rag_file = None
    selected_model_key_for_rag_info = session.get("selected_model", DEFAULT_MODEL)
    current_model_config_for_rag_info = AVAILABLE_MODELS.get(selected_model_key_for_rag_info)
    if current_model_config_for_rag_info and "rag_filename_getter" in current_model_config_for_rag_info:
        try:
            # get_processed_filename_openai fonksiyonu artık parametre almıyor
            current_rag_file = current_model_config_for_rag_info["rag_filename_getter"]()
        except Exception as e_rag_get:
            logger.warning(f"{current_model_config_for_rag_info['display_name']} için RAG dosya adı alınırken hata: {e_rag_get}")

    if request.method == "POST":
        action = request.form.get("action")
        logger.info(f"POST isteği alındı. Action: {action}, Form: {request.form}")

        if action == "toggle_chat":
            session["chat_open"] = not session["chat_open"]
            return redirect(url_for("index"))

        elif action == "change_model":
            new_model_selected = request.form.get("model")
            if new_model_selected and new_model_selected in AVAILABLE_MODELS:
                if session.get("selected_model") != new_model_selected:
                    session["selected_model"] = new_model_selected
                    new_model_display_name_base = AVAILABLE_MODELS[new_model_selected]['display_name']
                    flash(f"Model {new_model_display_name_base} olarak değiştirildi.", "success")
            else:
                flash(f"Geçersiz model seçimi: {new_model_selected}", "warning")
            return redirect(url_for("index"))

        elif action == "send_message":
            user_input_value = request.form.get("user_input", "").strip()
            if not user_input_value:
                flash("Boş mesaj gönderilemez.", "warning")
                return redirect(url_for("index"))

            chosen_model_key = session.get("selected_model", DEFAULT_MODEL)
            model_cfg = AVAILABLE_MODELS.get(chosen_model_key)

            reply = ""
            history = list(session.get("chat_history", []))
            history.append(("Sen", user_input_value))

            if model_cfg:
                try:
                    reply = model_cfg["func"](user_input_value)
                    history.append((model_cfg["history_name"], reply))

                    # --- Kalıcı kayıt (JSON + Airtable) ---
                    try:
                        logger.info(f"Sohbet kaydediliyor: model={chosen_model_key}")
                        memory_manager.save_conversation(chosen_model_key, user_input_value, reply)
                        logger.info("✅ Web sohbeti başarıyla kaydedildi")
                    except Exception as e_mem:
                        logger.error("Sohbet hafızaya kaydedilirken hata: %s", e_mem, exc_info=True)
                        flash("Sohbet geçmişi kaydedilirken bir sorun oluştu.", "warning")

                except Exception as e:
                    logger.error("Model (%s) çağrılırken hata: %s", chosen_model_key, e, exc_info=True)
                    err_msg = f"{model_cfg['history_name']} yanıt alınırken sorun: {str(e)}"
                    history.append(("Sistem", err_msg))
                    flash(err_msg, "danger")
            else:
                reply = "Seçilen model için yapılandırma bulunamadı."
                history.append(("Sistem", reply))
                flash("Kritik hata: Model yapılandırması eksik.", "danger")

            # son 20 mesaj
            session["chat_history"] = history[-20:]
            return redirect(url_for("index"))

        elif action == "upload_document":
            flash("Dosya yükleme devre dışı. Gerekli belgeler geliştirici tarafından önceden yüklendi.", "info")
            return redirect(url_for("index"))

        elif action == "clear_rag":
            target_model_key = session.get("selected_model", DEFAULT_MODEL)
            model_config_for_clear = AVAILABLE_MODELS.get(target_model_key)
            if not model_config_for_clear or "rag_clearer" not in model_config_for_clear:
                flash(f"{model_config_for_clear.get('display_name', 'Seçili model')} RAG temizlemeyi desteklemiyor.", "warning")
                return redirect(url_for("index"))
            try:
                # clear_rag_context_openai artık parametre almıyor
                message = model_config_for_clear["rag_clearer"]()
                flash(f"RAG bağlamı temizlendi: {message}", 'info')
            except Exception as e:
                logger.error(f"RAG bağlamı temizlenirken hata: {e}", exc_info=True)
                flash(f"RAG bağlamı temizlenirken bir hata oluştu: {str(e)}", 'danger')
            return redirect(url_for("index"))

    dynamic_available_models = {}
    for key, data in AVAILABLE_MODELS.items():
        dynamic_data = data.copy()
        rag_file_for_this_model = None
        if "rag_filename_getter" in data:
            try:
                # get_processed_filename_openai fonksiyonu artık parametre almıyor
                processed_files_str = data["rag_filename_getter"]()
                if processed_files_str and processed_files_str != str({'tr': None, 'en': None}):
                     rag_file_for_this_model = processed_files_str
            except Exception as e_rag_get:
                logger.warning(f"{data['display_name']} için RAG dosya adı alınırken hata: {e_rag_get}")

        display_name_suffix = f" (RAG: {rag_file_for_this_model})" if rag_file_for_this_model else " (RAG Kapalı)"
        base_display_name = data["display_name"].split(' (RAG')[0]
        dynamic_data["display_name"] = base_display_name + display_name_suffix
        dynamic_available_models[key] = dynamic_data

    return render_template(
        "index.html",
        chat_open=session.get("chat_open"),
        chat_history=session.get("chat_history", []),
        selected_model=session.get("selected_model"),
        available_models_dict=dynamic_available_models,
        current_rag_file_for_selected_model=current_rag_file,
        allowed_extensions_str=", ".join([f".{ext}" for ext in ALLOWED_EXTENSIONS])
    )

# --- WHATSAPP WEBHOOK ENTEGRASYONU ---

VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "varsayilan-dogrulama-kodu")
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.environ.get("PHONE_NUMBER_ID")
CATALOG_URL = "https://ads.ozgenplastik.com/assets/pdf/catalog/1238e9b6-fb8b-45cf-9f08-74f9e235f8d0.pdf"


@app.route("/api/whatsapp", methods=['GET', 'POST'])
def webhook_whatsapp():
    if request.method == 'GET':
        if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
            if not request.args.get("hub.verify_token") == VERIFY_TOKEN:
                return "Verification token mismatch", 403
            return request.args.get("hub.challenge"), 200
        return "Merhaba bu bir webhook URL'sidir.", 200

    elif request.method == 'POST':
        data = request.get_json()
        logger.info(f"Gelen WhatsApp verisi: {json.dumps(data, indent=2)}")

        # Gelen verinin doğru formatta olup olmadığını kontrol edelim
        if data.get("object") == "whatsapp_business_account":
            try:
                for entry in data.get("entry", []):
                    for change in entry.get("changes", []):
                        if "messages" in change.get("value", {}):
                            for message in change.get("value", {}).get("messages", []):
                                if message.get("type") == "image":
                                    from_number = message.get("from")
                                    reply_text = (
                                        "Fotoğraf üzerinden ürün tespiti şu an aktif değil. 🙏\n"
                                        "Ürünü katalogdan bulup ürün adını veya ürün kodunu yazarsanız hemen yardımcı olurum.\n\n"
                                        f"Katalog: {CATALOG_URL}\n\n"
                                        "Örn: “322” veya “Büyük rattan tabure”")
                                    send_whatsapp_message(from_number, reply_text)
                                    try:
                                        logger.info(f"WhatsApp resim mesajı kaydediliyor: {from_number}")
                                        memory_manager.save_conversation(
                                             f"WhatsApp-{from_number}",
                                            "[IMAGE]",
                                             reply_text)
                                        logger.info("✅ WhatsApp resim yönlendirmesi kaydedildi")
                                    except Exception as e_mem:
                                        logger.error(
                                            "WhatsApp resim yönlendirmesi kaydedilirken hata: %s",
                                            e_mem,
                                            exc_info=True)
                                    continue   
                                if message.get("type") == "text":
                                    from_number = message.get("from")
                                    msg_body = message.get("text", {}).get("body")
                                    logger.info(f"'{from_number}' numarasından mesaj: '{msg_body}'")
                                    msg_body = (msg_body or "").strip()
                                    if not msg_body:
                                        continue
                                    # 2) ÖDEME / IBAN SORUSU → OFİSE YÖNLENDİR (LLM'e gitme)
                                    if is_payment_question(msg_body):
                                        reply_text = (
                                            "Ödeme/IBAN bilgileri için güvenlik  nedeniyle buradan paylaşım yapamıyoruz.\n"
                                             "Lütfen ofisimizle iletişime geçin: +90 545 659 54 31")
                                        send_whatsapp_message(from_number, reply_text)
                                        try:
                                            memory_manager.save_conversation(f"WhatsApp-{from_number}", msg_body, reply_text)
                                        except Exception as e_mem:
                                            logger.error("WhatsApp ödeme yönlendirme kaydı hatası: %s", e_mem, exc_info=True)
                                        continue

                                    # Gelen mesajın dilini tespit et
                                    try:
                                        detected_lang = detect(msg_body)
                                        logger.info(f"Tespit edilen dil: {detected_lang}")
                                    except LangDetectException:
                                        detected_lang = 'tr' # Tespit edilemezse varsayılan Türkçe
                                        logger.warning("Dil tespit edilemedi, varsayılan 'tr' kullanılıyor.")

                                    # Sadece 'tr' ve 'en' dillerini destekliyoruz
                                    if detected_lang not in ['tr', 'en']:
                                        # Diğer diller için varsayılan olarak Türkçe kullan
                                        lang_to_use = 'tr'
                                        logger.warning(f"Desteklenmeyen dil ({detected_lang}) tespit edildi, varsayılan 'tr' kullanılıyor.")
                                    else:
                                        lang_to_use = detected_lang
                                    if detected_lang == "en":
                                        reply_text = ("For English support and export inquiries, please contact:\n"
                                        "export@ozgenplastik.com")
                                        send_whatsapp_message(from_number, reply_text)
                                        try: 
                                            memory_manager.save_conversation(f"WhatsApp-{from_number}", msg_body, reply_text)
                                        except Exception as e_mem:
                                            logger.error("WhatsApp export yönlendirme kaydı hatası: %s", e_mem, exc_info=True)
                                        continue    

                                    # Chatbot'umuzdan yanıt alalım
                                    bot_response = openai_main_func(msg_body)
                                    logger.info(f"Chatbot yanıtı ({lang_to_use}): {bot_response}")

                                    # --- WhatsApp sohbetini de kaydet (YENİ) ---
                                    try:
                                        logger.info(f"WhatsApp sohbeti kaydediliyor: {from_number}")
                                        memory_manager.save_conversation(f"WhatsApp-{from_number}", msg_body, bot_response)
                                        logger.info("✅ WhatsApp sohbeti başarıyla kaydedildi")
                                    except Exception as e_mem:
                                        logger.error("WhatsApp sohbeti kaydedilirken hata: %s", e_mem, exc_info=True)

                                    # WhatsApp'a yanıtı gönderelim
                                    send_whatsapp_message(from_number, bot_response)

                return "OK", 200
            except Exception as e:
                logger.error(f"Webhook işlenirken hata oluştu: {e}", exc_info=True)
                return "Internal Server Error", 500

        return "Not a WhatsApp Business Account event", 404

def is_payment_question(text: str) -> bool:
    if not text:
        return False
    t = text.lower()

    keywords = [
        "iban", "ıban",
        "hesap no", "hesap numara", "hesap numarası",
        "banka", "eft", "havale", "swift",
        "ödeme", "odeme", "payment", "pay",
        "kredi kart", "kredi kartı", "credit card", "card", "pos"
    ]
    return any(k in t for k in keywords)

def send_whatsapp_message(to_number, text_message):
    """Verilen numaraya metin mesajı gönderir."""
    if not all([WHATSAPP_TOKEN, PHONE_NUMBER_ID]):
        logger.error("WhatsApp için ortam değişkenleri (WHATSAPP_TOKEN, PHONE_NUMBER_ID) eksik.")
        return

    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": text_message},
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # HTTP hatalarında exception fırlat
        logger.info(f"WhatsApp'a mesaj başarıyla gönderildi. Status: {response.status_code}, Response: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"WhatsApp'a mesaj gönderilirken hata oluştu: {e}")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
