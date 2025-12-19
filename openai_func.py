# openai_func.py - Güncellenmiş versiyonu

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document 
import pandas as pd  
import os
import logging
import re
from langchain_core.pydantic_v1 import BaseModel, Field, conlist
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langdetect import detect, LangDetectException
from enum import Enum
import json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


vector_stores = {
    "tr": None,
    "en": None
}
processed_filenames = {
    "tr": None,
    "en": None
}

# --- YENİ: ÜRÜN KODU EŞLEME TABLOSU ---
product_code_mapping = {}  # {kod: ürün_adı} şeklinde eşleme tablosu

# --- YENİ: SATIŞ YAPILAN ÜRÜN KODLARI ---
VALID_PRODUCT_CODES = [
    "322", "321", "005", "210", "003", "004", "301", "201", "202", "203", 
    "305", "306", "191", "192", "193", "309", "308", "307", "067", "013", 
    "014", "017", "147", "232", "218", "181", "186", "183", "182", "060", "323"
]

try:
    chat_model_openai = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1000)
    embeddings_model_openai = OpenAIEmbeddings(model="text-embedding-3-small")
    logger.info("OpenAI Chat ve Embeddings modelleri başarıyla yüklendi.")
except Exception as e:
    logger.error(f"OpenAI modelleri başlatılırken hata: {e}")
    chat_model_openai = None
    embeddings_model_openai = None


def _load_excel_with_pandas(file_path: str) -> list[Document]:
    """Excel dosyasını yükler ve aynı zamanda ürün kodu eşleme tablosunu oluşturur."""
    global product_code_mapping
    
    try:
        sheets = pd.read_excel(file_path, sheet_name=None, dtype=str)
    except Exception as e:
        logger.error(f"pandas ile Excel okunamadı: {e}")
        raise
    
    documents: list[Document] = []
    cols_priority = [
        "Ürün İsmi", "Ürün Kodu", "Ürün Koliçi Miktarı", "Ürün Ölçüleri",
        "Ürün Ağırlığı", "Koli Hacmi", "Ürün Fiyatı TL olarak",
        "Ürün Websitesi Linki", "Trendyol Linki"
    ]
    
    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue
        df = df.fillna("")
        normalized_cols = {c.lower().strip(): c for c in df.columns}
        
        def pick(name):
            key = name.lower().strip()
            return normalized_cols.get(key)
        
        # Ürün kodu ve ürün ismi sütunlarını bul
        product_name_col = pick("Ürün İsmi") or pick("ürün ismi") or pick("product name")
        product_code_col = pick("Ürün Kodu") or pick("ürün kodu") or pick("product code")
        
        for idx, row in df.iterrows():
            parts = []
            
            # --- YENİ: ÜRÜN KODU EŞLEMESİNİ OLUŞTUR ---
            if product_code_col and product_name_col:
                code = str(row.get(product_code_col, "")).strip()
                name = str(row.get(product_name_col, "")).strip()
                if code and name and code in VALID_PRODUCT_CODES:
                    product_code_mapping[code] = name
                    logger.info(f"Kod eşlemesi eklendi: {code} -> {name}")
            
            # Belge içeriğini oluştur
            for c in cols_priority:
                c_real = pick(c)
                if c_real and str(row.get(c_real, "")).strip():
                    parts.append(f"{c}: {row[c_real]}")
            
            for c in df.columns:
                if c in cols_priority:
                    continue
                val = str(row.get(c, "")).strip()
                if val:
                    parts.append(f"{c}: {val}")
            
            if not parts:
                continue
                
            content = " | ".join(parts)
            documents.append(
                Document(
                    page_content=content,
                    metadata={"source": os.path.basename(file_path), "sheet": sheet_name, "row_index": int(idx)}
                )
            )
    
    logger.info(f"Toplam {len(product_code_mapping)} ürün kodu eşlemesi oluşturuldu: {product_code_mapping}")
    return documents


def load_and_process_document_openai(file_path, filename_original, lang='tr'):
    global vector_stores, processed_filenames
    if not chat_model_openai or not embeddings_model_openai:
        return False, "OpenAI modelleri yüklenemedi."
    logger.info(f"OpenAI RAG için '{filename_original}' ({lang}) işleniyor...")
    try:
        lower = filename_original.lower()
        if lower.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif lower.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
        elif lower.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
        elif lower.endswith((".xlsx", ".xls")):
            documents = _load_excel_with_pandas(file_path)
        else:
            return False, "Desteklenmeyen dosya türü."
        if not documents:
            return False, "Dosya boş."
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        if not texts:
            return False, "Metin ayrılamadı."
        
        if vector_stores.get(lang) is None:
            vector_stores[lang] = FAISS.from_documents(texts, embeddings_model_openai)
        else:
            vector_stores[lang].add_documents(texts)

        current_filenames = processed_filenames.get(lang, "")
        processed_filenames[lang] = f"{current_filenames}, {filename_original}" if current_filenames else filename_original
        
        return True, f"'{filename_original}' ({lang}) eklendi."
    except Exception as e:
        logger.error(f"RAG hata: {e}", exc_info=True)
        return False, str(e)


def get_processed_filename_openai():
    return str(processed_filenames)

def clear_rag_context_openai():
    global vector_stores, processed_filenames, product_code_mapping
    vector_stores = {"tr": None, "en": None}
    processed_filenames = {"tr": None, "en": None}
    product_code_mapping = {}  # Kod eşleme tablosunu da temizle
    return "Tüm RAG bağlamları (TR & EN) ve ürün kodu eşlemeleri temizlendi."

# --- NİYET TESPİTİ MODELLERİ (değişiklik yok) ---
class UserIntent(str, Enum):
    SINGLE_PRODUCT_QUERY = "tekli_ürün_sorgusu"
    MULTI_PRODUCT_QUERY = "çoklu_ürün_sorgusu"
    PRODUCT_CODE_QUERY = "ürün_kodu_sorgusu"
    CATEGORY_QUERY = "kategori_sorgusu"
    GENERAL_QUERY = "genel_soru"
    FRUSTRATION = "musteri_sikayeti"
    URGENCY = "aciliyet_bildirimi"
    NEGOTIATION = "pazarlik_talebi"
    UNKNOWN = "bilinmeyen_soru"

class IntentAnalysisResult(BaseModel):
    intent: UserIntent = Field(description="Kullanıcının niyetini belirle.")
    products: list[str] = Field(description="Kullanıcının sorduğu ürünlerin listesi. Tek ürün varsa tek elemanlı liste olur. Ürün yoksa boş liste olur.")
    product_codes: list[str] = Field(description="Kullanıcının sorduğu ürün kodlarının listesi. Ürün kodu yoksa boş liste olur.")
    category: str = Field(description="Eğer niyet 'kategori_sorgusu' ise, sorgulanan kategoriyi belirtir. Değilse boş bırak.")

class ProductInfo(BaseModel):
    product_name: str = Field(description="Ürünün tam ve doğru adı")
    product_code: str = Field(description="Ürünün kodu")
    box_quantity: int = Field(description="Ürünün bir kolisindeki adet miktarı, sayı olarak")
    price: str = Field(description="Ürünün fiyatı (örn: 110 TL)")
    dimensions: str = Field(description="Ürünün ölçüleri (örn: 27 x 47 x 41)")
    weight: str = Field(description="Ürünün ağırlığı (örn: 14,25 kg)")
    shop_link: str = Field(description="Ürünün toptan satış web sitesi linki (shop.ozgenplastik.com)")
    trendyol_link: str = Field(description="Ürünün Trendyol linki. Eğer yoksa, boş bir string '' döndür.")
    found: bool = Field(description="Eğer ürün bilgisi bağlamda bulunduysa True, bulunamadıysa False")

def extract_quantity_and_clean_query(text: str) -> tuple[int | None, str]:
    quantity_keywords = ["tane", "adet", "koli", "piece", "pieces", "unit", "units", "box", "boxes"]
    pattern = r'\b(\d+)\s*(' + '|'.join(quantity_keywords) + r')\b'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        quantity = int(match.group(1))
        cleaned_query = text[:match.start()] + text[match.end():]
        return quantity, cleaned_query.strip()
    
    match_end = re.search(r'(\d+)\s*$', text)
    if match_end:
        quantity = int(match_end.group(1))
        cleaned_query = text[:match_end.start()].strip()
        return quantity, cleaned_query

    return None, text

def extract_product_codes(text: str) -> list[str]:
    """Metinden geçerli ürün kodlarını çıkarır."""
    found_codes = []
    for code in VALID_PRODUCT_CODES:
        patterns = [
            rf'\b{code}\b',
            rf'#{code}\b',
            rf'{code}\s+kod',
            rf'kod\s*:?\s*{code}',
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                if code not in found_codes:
                    found_codes.append(code)
                break
    return found_codes

def detect_lang(text: str) -> str:
    try:
        lang = detect(text)
        if lang in ['tr', 'en']:
            return lang
        return 'tr'
    except LangDetectException:
        return 'tr'


def _get_intent_detection_chain(lang: str) -> any:
    """Kullanıcı niyetini tespit etmek için bir LangChain zinciri oluşturur."""
    parser = JsonOutputParser(pydantic_object=IntentAnalysisResult)
    
    prompt_text = {
        'tr': """
        Sen bir metin sınıflandırma uzmanısın. Görevin, kullanıcının yazdığı metnin 'niyetini' analiz etmek ve JSON formatında çıktı vermektir.
        
        Niyet Kategorileri:
        - `tekli_ürün_sorgusu`: Kullanıcı tek bir ürün ismi hakkında bilgi istiyor. (Örn: "büyük rattan tabure 10 tane", "3 katlı ayakkabılık fiyatı")
        - `çoklu_ürün_sorgusu`: Kullanıcı birden fazla ürün ismi hakkında bilgi istiyor. (Örn: "Rattan tabure ve 32 litre sele ne kadar?")
        - `ürün_kodu_sorgusu`: Kullanıcı ürün kodları ile ürün arıyor. (Örn: "013 kodlu ürün 500 tane", "322 kodu ne kadar?", "005 ürününden 100 adet")
        - `kategori_sorgusu`: Kullanıcı genel bir ürün kategorisini soruyor. (Örn: "Tüm seleleriniz neler?", "Plastik bardak var mı?")
        - `genel_soru`: Ürünle ilgili olmayan genel bir soru. (Örn: "Adresiniz nerede?", "Showroom açık mı?", "Merhaba")
        - `musteri_sikayeti`: Kullanıcı şikayet veya hayal kırıklığı belirtiyor. (Örn: "50 kere sordum!", "Neden anlamıyorsunuz?")
        - `aciliyet_bildirimi`: Kullanıcı acil bir durum belirtiyor. (Örn: "Hemen lazım", "Acil sipariş")
        - `pazarlik_talebi`: Kullanıcı indirim veya pazarlık yapmaya çalışıyor. (Örn: "İndirim yapar mısınız?", "Bu fiyattan olmaz")
        - `bilinmeyen_soru`: Yukarıdakilere uymayan herhangi bir şey.

        Geçerli Ürün Kodları: {valid_codes}

        Kurallar:
        1. Kullanıcının sorusunu analiz et ve en uygun `intent` kategorisini seç.
        2. Eğer `tekli_ürün_sorgusu` veya `çoklu_ürün_sorgusu` ise, sorgudaki ürün isimlerini `products` listesine ekle.
        3. Eğer `ürün_kodu_sorgusu` ise, bulunan geçerli ürün kodlarını `product_codes` listesine ekle.
        4. Eğer `kategori_sorgusu` ise, kategorinin adını `category` alanına yaz.
        5. Sadece ve sadece JSON çıktısı ver. Başka hiçbir metin ekleme.

        Kullanıcı Sorusu: {question}
        JSON Çıktısı:
        {format_instructions}
        """,
        'en': """
        You are a text classification expert. Your task is to analyze the user's intent from their message and provide the output in JSON format.

        Intent Categories:
        - `tekli_ürün_sorgusu`: The user is asking about a single product name. (e.g., "large rattan stool 10 pieces", "price of 3-tier shoe rack")
        - `çoklu_ürün_sorgusu`: The user is asking about multiple product names. (e.g., "How much for the rattan stool and the 32-liter basket?")
        - `ürün_kodu_sorgusu`: The user is searching for products using product codes. (e.g., "product code 013, 500 pieces", "how much is code 322?", "100 units of product 005")
        - `kategori_sorgusu`: The user is asking about a general product category. (e.g., "What are all your baskets?", "Do you have plastic cups?")
        - `genel_soru`: A general, non-product-related question. (e.g., "Where is your address?", "Is the showroom open?", "Hello")
        - `musteri_sikayeti`: The user is expressing frustration or a complaint. (e.g., "I've asked 50 times!", "Why don't you understand?")
        - `aciliyet_bildirimi`: The user is indicating urgency. (e.g., "I need it now", "Urgent order")
        - `pazarlik_talebi`: The user is trying to negotiate or ask for a discount. (e.g., "Can you give a discount?", "Not at this price")
        - `bilinmeyen_soru`: Anything that does not fit the above categories.
        
        Valid Product Codes: {valid_codes}
        
        Rules:
        1. Analyze the user's query and select the most appropriate `intent` category.
        2. If the intent is `tekli_ürün_sorgusu` or `çoklu_ürün_sorgusu`, add the product names from the query to the `products` list.
        3. If the intent is `ürün_kodu_sorgusu`, add the found valid product codes to the `product_codes` list.
        4. If the intent is `kategori_sorgusu`, write the name of the category in the `category` field.
        5. Provide only the JSON output. Do not add any other text.
        
        User Query: {question}
        JSON Output:
        {format_instructions}
        """
    }

    valid_codes_str = ", ".join(VALID_PRODUCT_CODES)
    prompt = ChatPromptTemplate.from_template(
        template=prompt_text[lang],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "valid_codes": valid_codes_str
        },
    )
    return prompt | chat_model_openai | parser


def openai_func(user_input):
    global vector_stores, chat_model_openai
    
    lang = detect_lang(user_input)
    vector_store = vector_stores.get(lang)

    if not chat_model_openai or not vector_store:
        error_message = {
            'tr': "Hata: Modeller veya dile özel RAG veritabanı hazır değil.",
            'en': "Error: Models or language-specific RAG database not ready."
        }
        return error_message[lang]

    # --- ADIM 1: NİYETİ TESPİT ET ---
    try:
        intent_chain = _get_intent_detection_chain(lang)
        intent_result = intent_chain.invoke({"question": user_input})
        intent = UserIntent(intent_result['intent'])
        logger.info(f"Tespit edilen niyet: {intent.value}, Detaylar: {intent_result}")
    except Exception as e:
        logger.error(f"Niyet tespiti başarısız: {e}", exc_info=True)
        intent = UserIntent.GENERAL_QUERY
        intent_result = {'products': [], 'product_codes': [], 'category': ''}

    retriever = vector_store.as_retriever()

    # --- ADIM 2: NİYETE GÖRE EYLEM SEÇ ---
    
    if intent in [UserIntent.SINGLE_PRODUCT_QUERY, UserIntent.MULTI_PRODUCT_QUERY]:
        return _handle_product_query(user_input, intent_result.get('products', []), retriever, lang)
    
    elif intent == UserIntent.PRODUCT_CODE_QUERY:
        return _handle_product_code_query(user_input, intent_result.get('product_codes', []), retriever, lang)
    
    elif intent == UserIntent.CATEGORY_QUERY:
        return _handle_category_query(user_input, intent_result.get('category', ''), retriever, lang)

    elif intent == UserIntent.FRUSTRATION:
        return _handle_frustration(lang)
        
    elif intent == UserIntent.URGENCY:
        return _handle_urgency(lang)

    elif intent == UserIntent.NEGOTIATION:
        return _handle_negotiation(lang)

    else: # GENERAL_QUERY veya UNKNOWN
        return _run_general_query(user_input, retriever, lang)


def _get_query_expansion_chain(lang: str):
    """Sorgu genişletme için bir LangChain zinciri oluşturur."""
    prompt_text = {
        'tr': """
        Sen bir e-ticaret arama uzmanısın. Görevin, bir ürün sorgusunu alıp o ürünü bulmak için kullanılabilecek 3 farklı alternatif arama terimi üretmektir.
        Çıktın sadece virgülle ayrılmış bir liste olmalı. Başka hiçbir açıklama ekleme.
        
        Örnek 1:
        Soru: büyük rattan tabure
        Cevap: büyük rattan tabure, rattan mobilya, büyük tabure
        
        Örnek 2:
        Soru: 3'lü ayakkabılık
        Cevap: 3 katlı ayakkabılık, ayakkabı dolabı, ayakkabı rafı
        
        Soru: {question}
        Cevap:
        """,
        'en': """
        You are an e-commerce search expert. Your task is to take a product query and generate 3 different alternative search terms that could be used to find that product.
        Your output should be only a comma-separated list. Do not add any other explanation.
        
        Example 1:
        Question: large rattan stool
        Answer: large rattan stool, rattan furniture, big stool
        
        Example 2:
        Question: small tank stool
        Answer: small tank stool, folding stool, small plastic stool
        
        Question: {question}
        Answer:
        """
    }
    prompt = ChatPromptTemplate.from_template(prompt_text[lang])
    return prompt | chat_model_openai | StrOutputParser()


def _handle_product_code_query(user_input: str, product_codes: list[str], retriever, lang: str):
    """
    ŞİMDİ YENİ YAKLIŞIM: Kod → Ürün Adı eşlemesi yap, sonra ürün adıyla devam et
    """
    global product_code_mapping
    
    # Eğer niyet tespit sistemi kod bulamadıysa, manual olarak da deneyelim
    if not product_codes:
        product_codes = extract_product_codes(user_input)
    
    if not product_codes:
        no_code_message = {
            'tr': f"Geçerli bir ürün kodu bulamadım. Satış yaptığımız ürün kodları: {', '.join(VALID_PRODUCT_CODES)}",
            'en': f"I couldn't find a valid product code. Our available product codes are: {', '.join(VALID_PRODUCT_CODES)}"
        }
        return no_code_message[lang]
    
    # Adet bilgisini çıkar
    requested_quantity, _ = extract_quantity_and_clean_query(user_input)
    
    all_responses = []
    
    for product_code in product_codes:
        # 1. ADIM: KODU ÜRÜN ADINA ÇEVİR
        product_name = product_code_mapping.get(product_code)
        
        if not product_name:
            not_found_message = {
                'tr': f"'{product_code}' kodlu ürün sistemimizde bulunamadı.",
                'en': f"Product with code '{product_code}' was not found in our system."
            }
            all_responses.append(not_found_message[lang])
            continue
        
        logger.info(f"Kod {product_code} → Ürün adı: {product_name}")
        
        # 2. ADIM: ARTIK ÜRÜN ADIYLA NORMAL ÜRÜN SORGUSU GİBİ İŞLE
        # Bu kısım artık _handle_product_query ile aynı mantığı kullanacak
        
        # Sorgu genişletme
        try:
            query_expansion_chain = _get_query_expansion_chain(lang)
            expanded_queries_str = query_expansion_chain.invoke({"question": product_name})
            search_queries = [q.strip() for q in expanded_queries_str.split(',')]
            logger.info(f"Genişletilmiş sorgular for '{product_name}': {search_queries}")
        except Exception as e:
            logger.error(f"Sorgu genişletme başarısız: {e}", exc_info=True)
            search_queries = [product_name]

        # RAG'den belge al
        all_retrieved_docs = []
        for query in search_queries:
            try:
                docs = retriever.invoke(query)
                all_retrieved_docs.extend(docs)
            except Exception as e:
                logger.error(f"'{query}' için RAG alımı başarısız: {e}")
        
        # Tekrarlanan belgeleri kaldır
        unique_docs = {doc.page_content: doc for doc in all_retrieved_docs}.values()
        
        if not unique_docs:
            not_found_message = {
                'tr': f"'{product_name}' (Kod: {product_code}) hakkında detaylı bilgi bulamadım.",
                'en': f"I could not find detailed information about '{product_name}' (Code: {product_code})."
            }
            all_responses.append(not_found_message[lang])
            continue
            
        context_for_chain = "\n\n".join([doc.page_content for doc in unique_docs])
        
        parser = JsonOutputParser(pydantic_object=ProductInfo)
        
        prompt_texts = {
            'tr': """
            Sen bir ürün bilgi çıkarım asistanısın. Görevin, sana verilen Ürün Bilgileri bağlamını ve kullanıcının aradığı ürün adını analiz ederek, istenen ürünle ilgili bilgileri JSON formatında doldurmaktır.
            Kullanıcının sorusundaki ürünü bağlamda ara. Eğer ürünü bulursan, bilgilerini JSON şemasına göre doldur ve `found` alanını `true` yap.
            Eğer kullanıcı tarafından sorulan ürünü bağlamda bulamazsan, tüm alanları boş bırakarak sadece `found` alanını `false` yap.
            Sadece ve sadece JSON çıktısı ver. Başka hiçbir metin ekleme.

            Bağlam: --- {context} ---
            Kullanıcının Aradığı Ürün: {question}
            JSON Çıktısı: {format_instructions}
            """,
            'en': """
            You are a product information extraction assistant. Your task is to analyze the provided Product Information context and the user's question, then fill in the requested product information in JSON format.
            Search for the product mentioned in the user's question within the context. If you find the product, fill in the fields according to the JSON schema and set the `found` field to `true`.
            If you cannot find the product, leave all fields blank and just set the `found` field to `false`.
            Provide only the JSON output. Do not add any other text.
            
            Context: --- {context} ---
            Product Searched by User: {question}
            JSON Output: {format_instructions}
            """
        }

        prompt = ChatPromptTemplate.from_template(
            template=prompt_texts[lang],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | chat_model_openai
            | parser
        )

        try:
            product_info = chain.invoke({"context": context_for_chain, "question": product_name})
        except Exception as e:
            logger.error(f"JSON zinciri hatası ({lang}): {e}", exc_info=True)
            continue

        if not product_info or not product_info.get('found'):
            not_found_message = {
                'tr': f"'{product_name}' (Kod: {product_code}) hakkında detaylı bilgi bulamadım.",
                'en': f"I could not find detailed information about '{product_name}' (Code: {product_code})."
            }
            all_responses.append(not_found_message[lang])
            continue

        # Dile göre dinamik yanıt oluşturma
        response_parts = [
            f"--- {product_info['product_name']} (Kod: {product_code}) ---",
            f"*{'Ürün Kodu' if lang == 'tr' else 'Product Code'}:* {product_info['product_code']}",
            f"*{'Fiyat' if lang == 'tr' else 'Price'}:* {product_info.get('price', 'N/A')}",
            f"*{'Koli İçi Miktarı' if lang == 'tr' else 'Box Quantity'}:* {product_info.get('box_quantity', 'N/A')}",
            f"*{'Ölçüler' if lang == 'tr' else 'Dimensions'}:* {product_info.get('dimensions', 'N/A')}",
            f"*{'Ağırlık' if lang == 'tr' else 'Weight'}:* {product_info.get('weight', 'N/A')}",
        ]

        try:
            box_quantity = int(product_info.get('box_quantity', 0))
        except (ValueError, TypeError):
            box_quantity = 0
            
        if requested_quantity is None:
            # Adet belirtilmemişse genel bilgi ver
            response_parts.append(f"\n*{'Satın Alma Bilgileri' if lang == 'tr' else 'Purchase Information'}*")
            if product_info.get('shop_link'):
                response_parts.append(f"{'Toptan satış için' if lang == 'tr' else 'For wholesale purchases'}: {product_info['shop_link']}")
            if product_info.get('trendyol_link'):
                response_parts.append(f"{'Perakende satış için' if lang == 'tr' else 'For retail purchases'}: {product_info['trendyol_link']}")
        elif requested_quantity < box_quantity:
            # Perakende mantığı
            if product_info.get('trendyol_link'):
                response_parts.append(f"\n*{'Perakende Alım (Trendyol)' if lang == 'tr' else 'Retail Purchase (Trendyol)'}*")
                response_parts.append(f"{'Bu adetteki alımlarınız için' if lang == 'tr' else 'For purchases of this quantity'}: {product_info['trendyol_link']}")
            else:
                response_parts.append(f"\n*{'Perakende Satış Yok' if lang == 'tr' else 'No Retail Sale'}*")
                response_parts.append(f"{'Bu ürün için online perakende satışımız mevcut değildir.' if lang == 'tr' else 'Online retail sales are not available for this product.'}")
        else: # Toptan
            response_parts.append(f"\n*{'Toptan Alım' if lang == 'tr' else 'Wholesale Purchase'}*")
            if product_info.get('shop_link'):
                response_parts.append(f"{'Web sitemizden sipariş verebilirsiniz' if lang == 'tr' else 'You can order from our website'}: {product_info['shop_link']}")
            response_parts.append(f"{'Detaylı bilgi için ofisimizle iletişime geçin' if lang == 'tr' else 'Contact our office for details'}: +90 545 659 54 31")

        all_responses.append("\n".join(response_parts))

    if not all_responses:
        return _run_general_query(user_input, retriever, lang)

    return "\n\n".join(all_responses)


def _handle_product_query(user_input: str, products: list[str], retriever, lang: str):
    """
    Bir veya daha fazla ürün sorgusunu işler, RAG'den bilgi alır ve yanıtı oluşturur.
    """
    if not products:
         # Niyet tespit zinciri ürün bulamadıysa, genel sorguya yönlendir.
        return _run_general_query(user_input, retriever, lang)

    all_responses = []
    
    # Her bir ürün için RAG işlemini tekrarla
    for product_name_from_intent in products:
        # --- YENİ: SORGULARI GENİŞLET ---
        try:
            query_expansion_chain = _get_query_expansion_chain(lang)
            expanded_queries_str = query_expansion_chain.invoke({"question": product_name_from_intent})
            search_queries = [q.strip() for q in expanded_queries_str.split(',')]
            logger.info(f"Genişletilmiş sorgular for '{product_name_from_intent}': {search_queries}")
        except Exception as e:
            logger.error(f"Sorgu genişletme başarısız: {e}", exc_info=True)
            search_queries = [product_name_from_intent] # Hata durumunda orijinal sorguyla devam et

        # --- YENİ: Genişletilmiş sorgularla RAG'den belge al ---
        all_retrieved_docs = []
        for query in search_queries:
            try:
                all_retrieved_docs.extend(retriever.invoke(query))
            except Exception as e:
                 logger.error(f"'{query}' için RAG alımı başarısız: {e}")
        
        # Tekrarlanan belgeleri kaldır
        unique_docs = {doc.page_content: doc for doc in all_retrieved_docs}.values()
        
        if not unique_docs:
            logger.warning(f"Genişletilmiş sorgular için hiçbir belge bulunamadı: {search_queries}")
            # Belge bulunamazsa bu ürünü atla
            continue
            
        context_for_chain = "\n\n".join([doc.page_content for doc in unique_docs])
        # --- / YENİ ---

        # Sorguyu, niyetteki ürün adı + orijinal sorgudaki adet bilgisi ile birleştir
        requested_quantity, _ = extract_quantity_and_clean_query(user_input)
        
        # RAG için daha iyi sonuç almak adına hem niyetten gelen ürün adını hem de orijinal sorguyu kullanabiliriz.
        # Şimdilik daha basit bir yaklaşımla, niyetten gelen ürün adını kullanalım.
        query_for_rag = product_name_from_intent

        parser = JsonOutputParser(pydantic_object=ProductInfo)
        
        prompt_texts = {
            'tr': """
            Sen bir ürün bilgi çıkarım asistanısın. Görevin, sana verilen Ürün Bilgileri bağlamını ve kullanıcı sorusunu analiz ederek, istenen ürünle ilgili bilgileri JSON formatında doldurmaktır.
            Kullanıcının sorusundaki ürünü bağlamda ara. Eğer ürünü bulursan, bilgilerini JSON şemasına göre doldur ve `found` alanını `true` yap.
            Eğer kullanıcı tarafından sorulan ürünü bağlamda bulamazsan, tüm alanları boş bırakarak sadece `found` alanını `false` yap.
            Sadece ve sadece JSON çıktısı ver. Başka hiçbir metin ekleme.

            Bağlam: --- {context} ---
            Kullanıcının Aradığı Ürün: {question}
            JSON Çıktısı: {format_instructions}
            """,
            'en': """
            You are a product information extraction assistant. Your task is to analyze the provided Product Information context and the user's question, then fill in the requested product information in JSON format.
            Search for the product mentioned in the user's question within the context. If you find the product, fill in the fields according to the JSON schema and set the `found` field to `true`.
            If you cannot find the product, leave all fields blank and just set the `found` field to `false`.
            Provide only the JSON output. Do not add any other text.
            
            Context: --- {context} ---
            Product Searched by User: {question}
            JSON Output: {format_instructions}
            """
        }

        prompt = ChatPromptTemplate.from_template(
            template=prompt_texts[lang],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | chat_model_openai
            | parser
        )

        try:
            # Zincire artık retriever'ı değil, manuel olarak birleştirdiğimiz context'i veriyoruz.
            product_info = chain.invoke({"context": context_for_chain, "question": product_name_from_intent})
        except Exception as e:
            logger.error(f"JSON zinciri hatası ({lang}): {e}", exc_info=True)
            # Bir ürün için hata olursa atla ve devam et
            continue

        if not product_info or not product_info.get('found'):
            not_found_message = {
                'tr': f"'{product_name_from_intent}' ürünü hakkında bilgi bulamadım.",
                'en': f"I could not find information about the product '{product_name_from_intent}'."
            }
            all_responses.append(not_found_message[lang])
            continue

        # Dile göre dinamik yanıt oluşturma
        response_parts = [
            f"--- {product_info['product_name']} ---",
            f"*{'Ürün Kodu' if lang == 'tr' else 'Product Code'}:* {product_info['product_code']}",
            f"*{'Fiyat' if lang == 'tr' else 'Price'}:* {product_info.get('price', 'N/A')}",
            f"*{'Koli İçi Miktarı' if lang == 'tr' else 'Box Quantity'}:* {product_info.get('box_quantity', 'N/A')}",
            f"*{'Ölçüler' if lang == 'tr' else 'Dimensions'}:* {product_info.get('dimensions', 'N/A')}",
            f"*{'Ağırlık' if lang == 'tr' else 'Weight'}:* {product_info.get('weight', 'N/A')}",
        ]

        try:
            box_quantity = int(product_info.get('box_quantity', 0))
        except (ValueError, TypeError):
            box_quantity = 0
            
        if requested_quantity is None:
            # Adet belirtilmemişse genel bilgi ver
            pass
        elif requested_quantity < box_quantity:
            # Perakende mantığı
            if product_info.get('trendyol_link'):
                response_parts.append(f"\n*{'Perakende Alım (Trendyol)' if lang == 'tr' else 'Retail Purchase (Trendyol)'}*")
                response_parts.append(f"{'Bu adetteki alımlarınız için' if lang == 'tr' else 'For purchases of this quantity'}: {product_info['trendyol_link']}")
            else:
                response_parts.append(f"\n*{'Perakende Satış Yok' if lang == 'tr' else 'No Retail Sale'}*")
                response_parts.append(f"{'Bu ürün için online perakende satışımız mevcut değildir.' if lang == 'tr' else 'Online retail sales are not available for this product.'}")
        else: # Toptan
            response_parts.append(f"\n*{'Toptan Alım' if lang == 'tr' else 'Wholesale Purchase'}*")
            if product_info.get('shop_link'):
                response_parts.append(f"{'Web sitemizden sipariş verebilirsiniz' if lang == 'tr' else 'You can order from our website'}: {product_info['shop_link']}")
            response_parts.append(f"{'Detaylı bilgi için ofisimizle iletişime geçin' if lang == 'tr' else 'Contact our office for details'}: +90 545 659 54 31")

        all_responses.append("\n".join(response_parts))

    if not all_responses:
         # Eğer hiçbir ürün bulunamazsa genel bir mesaj döndür
        return _run_general_query(user_input, retriever, lang)

    return "\n\n".join(all_responses)

def _handle_category_query(user_input: str, category: str, retriever, lang: str):
    """Kategori sorgularını işler."""
    search_query = category or user_input
    
    prompt_text = {
        'tr': """
        Sen bir ürün listeleme asistanısın. Sana verilen bağlamdaki bilgileri kullanarak, kullanıcının sorduğu kategoriye ait TÜM ürünleri listele.
        Her ürün için Ürün İsmi ve varsa Websitesi Linkini yaz. Başka detay verme.
        Eğer kategoriye uygun ürün bulamazsan, "Bu kategoride ürün bulamadım." de.

        Bağlam: --- {context} ---
        Kullanıcının Sorguladığı Kategori: {question}
        Ürün Listesi:
        """,
        'en': """
        You are a product listing assistant. Using the information in the provided context, list ALL products belonging to the category the user asked for.
        For each product, write the Product Name and its Website Link if available. Do not provide other details.
        If you cannot find any products for the category, say "I could not find any products in this category."

        Context: --- {context} ---
        Category Queried by User: {question}
        Product List:
        """
    }
    
    prompt = ChatPromptTemplate.from_template(prompt_text[lang])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | chat_model_openai
        | StrOutputParser()
    )
    return chain.invoke(search_query)

def _handle_frustration(lang: str):
    responses = {
        'tr': "Yaşadığınız aksaklık için çok üzgünüm. Talebinizi doğru bir şekilde anladığımdan emin olmak adına, lütfen isteğinizi tek bir mesajda, ürün adı ve adet belirterek tekrar yazar mısınız? Size hemen yardımcı olacağım.",
        'en': "I am very sorry for the inconvenience you've experienced. To ensure I understand your request correctly, could you please rephrase it in a single message, specifying the product name and quantity? I will assist you immediately."
    }
    return responses[lang]

def _handle_urgency(lang: str):
    responses = {
        'tr': "Siparişinizin aciliyetini anlıyorum. Standart teslimat süreçlerimiz hakkında bilgi almak ve olası daha hızlı teslimat seçeneklerini görüşmek için lütfen doğrudan ofisimizle iletişime geçin: +90 545 659 54 31",
        'en': "I understand the urgency of your order. To inquire about our standard delivery processes and discuss potential faster delivery options, please contact our office directly: +90 545 659 54 31"
    }
    return responses[lang]

def _handle_negotiation(lang: str):
    responses = {
        'tr': "Fiyatlarımız standarttır. Ancak, çok yüksek adetli toptan alımlarınız için özel iskontoları görüşmek üzere satış departmanımızla iletişime geçebilirsiniz: +90 545 659 54 31",
        'en': "Our prices are standard. However, for very large wholesale purchases, you can contact our sales department to discuss potential special discounts: +90 545 659 54 31"
    }
    return responses[lang]


def _run_general_query(user_input, retriever, lang):
    """Genel konular için standart RAG zincirini çalıştırır. (Daha önce 'run_general_query' idi)"""
    global chat_model_openai
    
    prompt_texts = {
        'tr': """
        Sen Özgen Plastik için bir müşteri hizmetleri asistanısın. Görevin, SADECE sana verilen bağlamı kullanarak kullanıcının sorusuna net ve kısa bir cevap vermektir.

        **KRİTİK KURALLAR:**
        1. **Kullanıcının "Soru"su ile aynı dilde cevap ver.** Soru Türkçe ise, cevabın MUTLAKA Türkçe olmalı.
        2. **Özetle, kopyala-yapıştır yapma.** Bağlamı oku ve doğal bir cevap oluştur. Bağlamdaki diğer soruları (örneğin SSS'deki gibi) cevabına KESİNLİKLE ekleme.
        3. **Spesifik ol.** Sadece kullanıcının sorusunu cevapla. Konuyla ilgisi olmayan bilgileri bağlamdan alma. Kullanıcı ödeme yöntemlerini soruyorsa, sadece ödeme yöntemlerini anlat.
        4. Eğer cevap bağlamda yoksa, sadece bu konuda bilgi bulamadığını belirt.

        Bağlam:
        ---
        {context}
        ---

        Soru: {question}
        Cevap:
        """,
        'en': """
        You are a customer service assistant for Ozgen Plastik. Your task is to provide a clear and concise answer to the user's question based ONLY on the provided context.

        **CRITICAL RULES:**
        1. **Answer in the same language as the "Question".** If the question is in English, your answer MUST be in English. **This is true even if the entire context provided is in Turkish.** In that case, you must first translate the relevant information from the context to English and then formulate your answer.
        2. **Synthesize, do not copy-paste.** Read the context and formulate a natural answer in your own words. Do not repeat other questions from the context (like in a FAQ).
        3. **Be specific.** Directly answer the user's question. Do not include irrelevant information from the context.
        4. If the answer is not in the context, simply state that you could not find the information.

        Context:
        ---
        {context}
        ---

        Question: {question}
        Answer:
        """
    }
    
    general_prompt = ChatPromptTemplate.from_template(prompt_texts[lang])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    general_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | general_prompt
        | chat_model_openai
        | StrOutputParser()
    )
    try:
        answer = general_chain.invoke(user_input)
        if not answer or "bilgi bulamadım" in answer.lower() or "not find information" in answer.lower() or len(answer.strip()) < 15:
             error_message = {
                 'tr': "Üzgünüm, bu konuda yeterli bilgi bulamadım. Detaylı bilgi için lütfen ofisimizle iletişime geçin.\n- Ofis Telefonu: +90 545 659 54 31",
                 'en': "I'm sorry, I couldn't find enough information on this topic. For detailed information, please contact our office.\n- Office Phone: +90 545 659 54 31"
             }
             return error_message[lang]
        return answer
    except Exception as e:
        logger.error(f"Genel RAG zinciri hatası ({lang}): {e}", exc_info=True)
        error_message = {
            'tr': "Sorunuzu yanıtlarken bir hata oluştu.",
            'en': "An error occurred while answering your question."
        }
        return error_message[lang]
