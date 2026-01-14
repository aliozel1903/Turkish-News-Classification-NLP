import pandas as pd
import re
import string
import numpy as np
import torch
import random  # rastgele model secmek icin 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
import warnings

# Terminal kirli gozukmesin diye uyarilari kapattim
warnings.filterwarnings("ignore")



#csv dosyam
df = pd.read_csv('VeriSeti_Ali.csv')

# Sutun isimleri karisik gelmesin diye duzelttim
df = df.rename(columns={'HABERLER': 'metin', 'ETIKET': 'sinif'})

# Sadece bu 3 konuyla ilgileniyorum, digerlerini eledim
hedef_siniflar = ['Ekonomi', 'Spor', 'Teknoloji']
df = df[df['sinif'].isin(hedef_siniflar)]

# Her siniftan esit sayida (500) haber aldim ki model yanli olmasin
df = df.groupby('sinif').head(500)

# Veriyi karistiriyorum (shuffle) - random_state 61 yaptim sonuclar sabit kalsin diye
df = df.sample(frac=1, random_state=61).reset_index(drop=True)
df = df.dropna() # Bos satir varsa ucuruyorum

print(f"Veri seti hazir. Toplam satir: {len(df)}")
print("-" * 30)



# Bunlar gereksiz kelimeler (baglaclar vs.), anlam ifade etmiyorlar
stop_words = set([
    've', 'ile', 'bir', 'bu', 'şu', 'o', 'için', 'diye', 'da', 'de', 
    'mi', 'mu', 'ama', 'fakat', 'lakin', 'ki', 'daha', 'en', 'çok', 'gibi'
])

def metin_temizle(metin):
    # Once kucuk harfe ceviriyorum, Turkce karakter sorununu cozmek icin replace yaptim
    metin = metin.replace("İ", "i").replace("I", "ı").lower()
    # Noktalama isaretlerini kaldiriyorum
    metin = metin.translate(str.maketrans('', '', string.punctuation))
    # Sayilari siliyorum, haberin konusuyla ilgisi yok cunku
    metin = re.sub(r'\d+', '', metin)
    
    kelimeler = metin.split()
    # Stop words listesindekileri atiyorum
    temiz_kelimeler = [kelime for kelime in kelimeler if kelime not in stop_words]
    return " ".join(temiz_kelimeler)

# Temizligi tum veri setine uyguluyorum
df['temiz_metin'] = df['metin'].apply(metin_temizle)



# Bilgisayar kelimeden anlamaz, sayiya ceviriyoruz burada

print("Metinler sayilara cevriliyor...")

# Yontem A: Bag of words (Saydirma mantigi) - En cok gecen 2500 kelimeye bakiyoruz
bow_vectorizer = CountVectorizer(max_features=2500)
X_bow = bow_vectorizer.fit_transform(df['temiz_metin']).toarray()

# Yontem B: TF-IDF (Kelimenin onem derecesine gore puanlama)
tfidf_vectorizer = TfidfVectorizer(max_features=2500)
X_tfidf = tfidf_vectorizer.fit_transform(df['temiz_metin']).toarray()

# Yontem C: N-Grams (Kelimeleri ikili gruplar halinde alma) - Baglam icin onemli
ngram_vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=2500)
X_ngram = ngram_vectorizer.fit_transform(df['temiz_metin']).toarray()

# Yontem D: Word2Vec (Kelimelerin anlam iliskisi)
tokenized_data = df['temiz_metin'].apply(lambda x: x.split())
w2v_model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=2, workers=4)

# Her cumlenin ortalama vektorunu almam lazim
def get_average_word2vec(tokens, model, vector_size):
    vector_list = [model.wv[word] for word in tokens if word in model.wv]
    if len(vector_list) == 0:
        return np.zeros(vector_size)
    return np.mean(vector_list, axis=0)

X_w2v = np.array([get_average_word2vec(tokens, w2v_model, 100) for tokens in tokenized_data])

# Yontem E: BERT (Google'in gelismis modeli - biraz yavas calisiyor ama cok iyi)
print("BERT modeli yukleniyor (biraz surebilir)...")
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_bert_embeddings(text_list):
    embeddings = []
    for i, text in enumerate(text_list):
        if i % 250 == 0 and i > 0: 
            print(f"  -> {i} tanesi islendi...") # Islem uzun surerse gorelim diye
            
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_embedding.flatten())
    return np.array(embeddings)

X_bert = get_bert_embeddings(df['temiz_metin'])




print("\nSimdi modellere ders calistiriyorum...")
print("-" * 50)

# Siniflari (Spor, Ekonomi) 0,1,2 diye kodluyorum
le = LabelEncoder()
y = le.fit_transform(df['sinif'])

# Hepsini sirayla deneyecegim liste
methods = [
    ("Bag-of-Words", X_bow),
    ("TF-IDF", X_tfidf),
    ("N-Grams", X_ngram),
    ("Word2Vec", X_w2v),
    ("BERT", X_bert)
]

trained_models = {} # Egitilenleri buraya kaydedecegim

def train_and_evaluate(X, y, method_name):
    # Verinin %25'ini sakliyorum ki model ezberlemesin, sinav yapalim
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=61)
    
    # Model olarak Lojistik Regresyon kullaniyorum. C=2.0 yaptim biraz daha kati olsun diye.
    clf = LogisticRegression(C=2.0, max_iter=2000)
    clf.fit(X_train, y_train) # Iste egitim burada yapiliyor
    
    # Test verisiyle tahmin yaptirip notunu veriyorum
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Yontem: {method_name} -> Basari Orani: %{acc*100:.2f}")
    
    return clf

for name, X_matrix in methods:
    trained_models[name] = train_and_evaluate(X_matrix, y, name)

print("-" * 50)
print(df.head(10))


# kullanicidan veri alip tahmin ettigimiz yer

def tahmin_et(metin):
    try:
        if not metin or len(metin.strip()) < 5:
            print("Hocam cok kisa yazdiniz, biraz daha uzun cumle lazim.\n")
            return

        temiz = metin_temizle(metin)
        
        # Odevde farkli yontemler istenmisti, her seferinde rastgele birini seciyoruz
        mevcut_modeller = ["Bag-of-Words", "TF-IDF", "N-Grams", "Word2Vec", "BERT"]
        secilen_model_adi = random.choice(mevcut_modeller)
        
        clf = trained_models[secilen_model_adi]
        vektor = None
        
        # Secilen modele gore metni uygun formata ceviriyoruz
        if secilen_model_adi == "Bag-of-Words":
            vektor = bow_vectorizer.transform([temiz]).toarray()
        elif secilen_model_adi == "TF-IDF":
            vektor = tfidf_vectorizer.transform([temiz]).toarray()
        elif secilen_model_adi == "N-Grams":
            vektor = ngram_vectorizer.transform([temiz]).toarray()
        elif secilen_model_adi == "Word2Vec":
            tokens = temiz.split()
            vektor = np.array([get_average_word2vec(tokens, w2v_model, 100)])
        elif secilen_model_adi == "BERT":
            vektor = get_bert_embeddings([temiz])

        # Tahmin ani
        tahmin_index = clf.predict(vektor)[0]
        tahmin_sinif = le.inverse_transform([tahmin_index])[0]
        
        # Guven skorunu hesapliyoruz
        guven = clf.predict_proba(vektor)[0][tahmin_index]
        
        print("-" * 50)
        print(f"GIRILEN HABER: {metin[:100]}...") 
        print(f"🎲 KURA SONUCU SECILEN MODEL: {secilen_model_adi}") 
        print(f"🎯 TAHMIN: {tahmin_sinif.upper()}")
        print(f"📊 GUVEN SKORU: %{guven*100:.2f}")
        print("-" * 50 + "\n")
        
    except Exception as e:
        print(f"Bi hata oldu hocam: {e}")



print("\n" + "="*50)
print("SISTEM HAZIR! (Cikmak icin 'q' yazip enter'a basabilirsiniz)")
print("="*50)

while True:
    kullanici_girdisi = input("Haber metnini yapistirin: ")
    
    if kullanici_girdisi.lower() == 'q':
        print("Gorusmek uzere...")
        break
        
    tahmin_et(kullanici_girdisi)