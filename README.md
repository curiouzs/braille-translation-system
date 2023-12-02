# braille-translation-system
## text extraction
```py
import pytesseract
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import fitz
from PIL import Image
# pdf file 
pdf_document = 'image/sample_file.pdf
pdf_file = fitz.open(pdf_document)
for page_number in range(pdf_file.page_count):
    page = pdf_file[page_number]
    img = page.get_pixmap()
    text = pytesseract.image_to_string(Image.frombytes("RGB", (img.width, img.height),
 img.samples), lang= "tam+Telugu+Kannada+en")
    print(f"Page {page_number + 1}:\n{text}\n")
pdf_file.close()
# DOCX file
from docx import Document
docx_document = 'image/ieee.docx'
doc = Document(docx_document)
text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
print(text)
# img file
import cv2
img_cv = cv2.imread('image/pana.jpg')
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
ocr_res = pytesseract.image_to_string(img_rgb, lang='Telugu+Kannada+en')
print(ocr_res)

```

## model 
```py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv("EnglishBraille.csv")
english_list = [english for english in df['english ']]
braille_list = [braille for braille in df['braille']]
#Applying tokenizer
english_list = [str(english) for english in english_list]
tokenizer_eng = Tokenizer()
tokenizer_eng.fit_on_texts(english_list)
eng_seq = tokenizer_eng.texts_to_sequences(english_list)
# Tokenize braille
tokenizer_br = Tokenizer()
tokenizer_br.fit_on_texts(braille_list)
br_seq = tokenizer_br.texts_to_sequences(braille_list)
vocab_size_eng = len(tokenizer_eng.word_index) + 1
vocab_size_br = len(tokenizer_br.word_index) + 1
max_length = max(len(seq) for seq in eng_seq + br_seq)
eng_seq_padded = pad_sequences(eng_seq, maxlen=max_length, padding='post')
br_seq_padded = pad_sequences(br_seq, maxlen=max_length, padding='post')
embedding_dim = 256
units = 512
# Encoder
encoder_inputs = Input(shape=(max_length,))
enc_emb = Embedding(input_dim=vocab_size_eng, output_dim=embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]
decoder_inputs = Input(shape=(max_length,))
dec_emb_layer = Embedding(input_dim=vocab_size_br, output_dim=embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_size_br, activation='softmax')
output = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
X_train, X_val, y_train, y_val = train_test_split(eng_seq_padded, br_seq_padded, test_size=0.2)
model.fit([X_train, X_train], y_train, validation_data=([X_val, X_val], y_val), epochs=2, batch_size=64)
model.save("seq2seq_translation_v4.h5")
def translate_sentence(sentence):
    seq = tokenizer_eng.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    translated = np.argmax(model.predict([padded, padded]), axis=-1)

    translated_sentence = []
    for i in translated[0]:
        if i in tokenizer_br.index_word:
            translated_sentence.append(tokenizer_br.index_word[i])
        else:
            translated_sentence.append(' ')  # Token inconnu si l'indice n'est pas trouvé dans le tokenizer

    return ' '.join(translated_sentence)
input_sentence = "this is an acknowledgement to appoint Lokesh as business manager"

print(f"Input: {input_sentence}")
translated_sentence = translate_sentence(input_sentence)
print(translated_sentence, end = ' ')

```
