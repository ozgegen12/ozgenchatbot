# --- START OF FILE file_load.py ---
import json
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from dotenv import load_dotenv

# .env'i proje kökünden ve override=True ile yükle (env çakışmalarını önler)
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

MEMORY_DIR = "memory"
MEMORY_FILE = os.path.join(MEMORY_DIR, "memory.json")

# pyairtable (Airtable resmi Python kütüphanesi)
# pip install pyairtable python-dotenv
try:
    from pyairtable import Api
except Exception:
    Api = None


def _as_bool(s: str, default: bool = False) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in {"true", "1", "yes", "on"}


class AirtableSink:
    """
    pyairtable ile Airtable'a kayıt atar.

    Gerekli .env:
      AIRTABLE_ENABLED=true/false
      AIRTABLE_API_KEY=pat_xxx
      AIRTABLE_BASE_ID=appxxxx
      AIRTABLE_TABLE_NAME=tblaXXXX  (veya "Table 1")

    Kolon adları (tablondakiyle aynı olmalı):
      AIRTABLE_FIELD_MODEL_NAME=Model
      AIRTABLE_FIELD_USER_INPUT_NAME=UserInput
      AIRTABLE_FIELD_BOT_RESP_NAME=BotResponse
      AIRTABLE_FIELD_TIMESTAMP_NAME=timestamp

    timestamp alanı 'Date' ise gün formatı (YYYY-MM-DD) gönderilir.
    """
    def __init__(self) -> None:  # ← DÜZELTME: _init_ yerine __init__
        self.enabled = _as_bool(os.getenv("AIRTABLE_ENABLED"), default=False)
        self.api_key = os.getenv("AIRTABLE_API_KEY", "").strip()
        self.base_id = os.getenv("AIRTABLE_BASE_ID", "").strip()
        self.table_name = os.getenv("AIRTABLE_TABLE_NAME", "").strip()

        # Alan adları — varsayılanları senin tablo düzenin
        self.f_model = os.getenv("AIRTABLE_FIELD_MODEL_NAME", "Model").strip()
        self.f_user  = os.getenv("AIRTABLE_FIELD_USER_INPUT_NAME", "UserInput").strip()
        self.f_bot   = os.getenv("AIRTABLE_FIELD_BOT_RESP_NAME", "BotResponse").strip()
        self.f_time  = os.getenv("AIRTABLE_FIELD_TIMESTAMP_NAME", "timestamp").strip()

        self.table = None

        if not self.enabled:
            print("ℹ Airtable kapalı (AIRTABLE_ENABLED=false). Sadece JSON'a yazılacak.")
            return

        if Api is None:
            print("⚠ pyairtable bulunamadı. pip install pyairtable kurun. Airtable devre dışı.")
            self.enabled = False
            return

        if not (self.api_key and self.base_id and self.table_name):
            print("⚠ Airtable ENV eksik (API_KEY/BASE_ID/TABLE_NAME). Airtable devre dışı.")
            self.enabled = False
            return

        try:
            api = Api(self.api_key)
            self.table = api.table(self.base_id, self.table_name)
            print(f"ℹ Airtable hedefi: base={self.base_id}, table={self.table_name}")
        except Exception as e:
            print(f"⚠ Airtable başlatma hatası: {e}")
            self.enabled = False

    @staticmethod
    def _date_for_airtable() -> str:
        """Airtable 'Date' alanı için UTC gün (YYYY-MM-DD)."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def save(self, record: Dict[str, str]) -> bool:
        if not (self.enabled and self.table):
            return False

        # Airtable'a gidecek alanlar — Model tek satır metin, diğerleri long text, timestamp date
        fields = {
            self.f_model: record.get("model", ""),          # örn: "GPT-4o-mini"
            self.f_user:  record.get("user_input", ""),
            self.f_bot:   record.get("bot_response", ""),
            self.f_time:  self._date_for_airtable(),
        }

        try:
            self.table.create(fields, typecast=True)
            return True
        except Exception as e:
            print(f"❌ Airtable create hatası: {e}")
            return False


class MemoryManager:
    """
    1) JSON'a kayıt (primary)
    2) AIRTABLE_ENABLED=true ise Airtable'a da kayıt
    """
    def __init__(self) -> None:  # ← DÜZELTME: _init_ yerine __init__
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self._init_local_store()
        self.airtable = AirtableSink()

    def _init_local_store(self) -> None:
        if not os.path.exists(MEMORY_FILE):
            self._atomic_write_json([])

    def _atomic_write_json(self, data: List[Dict]) -> None:
        """JSON dosyasına atomik yaz (kısmi yazımı önler)."""
        fd, tmp_path = tempfile.mkstemp(dir=MEMORY_DIR, prefix="memory_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            os.replace(tmp_path, MEMORY_FILE)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

    def save_conversation(self, model: str, user_input: str, bot_response: str) -> None:
        # JSON için tam zaman (okunabilir)
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "user_input": user_input,
            "bot_response": bot_response,
        }

        # 1) JSON
        data = self.load_memory()
        data.append(record)
        self._atomic_write_json(data)
        print("✅ JSON'a sohbet kaydedildi.")

        # 2) Airtable
        if self.airtable.enabled:
            if self.airtable.save(record):
                print("✅ Airtable'a sohbet kaydedildi.")
            else:
                print("⚠ Airtable kaydı başarısız (lokalde JSON duruyor).")

    def load_memory(self) -> List[Dict]:
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def get_all_conversations(self) -> List[Tuple[int, str, str, str, str]]:
        rows: List[Tuple[int, str, str, str, str]] = []
        for i, rec in enumerate(self.load_memory(), start=1):
            rows.append((
                i,
                rec.get("model", ""),
                rec.get("user_input", ""),
                rec.get("bot_response", ""),
                rec.get("timestamp", ""),
            ))
        return rows
# --- END OF FILE file_load.py ---
