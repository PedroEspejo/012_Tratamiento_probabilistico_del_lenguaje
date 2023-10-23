import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Elemento 1: Modelado de Incertidumbre
# Suponemos datos de texto con cierta variabilidad.
texts = ["Este es un buen producto.", "No estoy seguro acerca de esto.", "Me encanta completamente."]

# Elemento 4: Aprendizaje Automático Probabilístico
# Creamos un modelo de clasificación de sentimiento.
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

# Elemento 5: Toma de Decisiones Probabilística
# Transformamos los textos en secuencias de números.
sequences = tokenizer.texts_to_sequences(texts)
sequences_padded = pad_sequences(sequences, maxlen=6, padding='post', truncating='post')

# Elemento 6: Robustez y Adaptabilidad
# Elemento 7: Control y Toma de Decisiones

# Elemento 8: Redes Neuronales Profundas
# Creamos un modelo de red neuronal simple.
model = keras.Sequential([
    keras.layers.Embedding(len(word_index) + 1, 16, input_length=6),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation='sigmoid')
])

# Elemento 1: Modelado de Incertidumbre

# Elemento 2: Redes Bayesianas (Opcional)
# Elemento 3: Inferencia Probabilística (Opcional)

# Compilamos el modelo.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Elemento 4: Aprendizaje Automático Probabilístico

# Elemento 5: Toma de Decisiones Probabilística
# Elemento 6: Robustez y Adaptabilidad
# Elemento 7: Control y Toma de Decisiones

# Elemento 8: Redes Neuronales Profundas (Opcional)

# Entrenamos el modelo.
labels = np.array([1, 0, 1])  # Etiquetas de sentimiento (positivo o negativo)
model.fit(sequences_padded, labels, epochs=10)

# Elemento 5: Toma de Decisiones Probabilística

# Realizamos predicciones en nuevos textos.
new_texts = ["Este es un producto excelente.", "No me gusta en absoluto."]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_sequences_padded = pad_sequences(new_sequences, maxlen=6, padding='post', truncating='post')
predictions = model.predict(new_sequences_padded)

# Elemento 6: Robustez y Adaptabilidad
# Elemento 7: Control y Toma de Decisiones

# Elemento 8: Redes Neuronales Profundas (Opcional)

# Imprimimos las predicciones.
for i, text in enumerate(new_texts):
    sentiment = "positivo" if predictions[i] > 0.5 else "negativo"
    print(f"Texto: {text}, Sentimiento: {sentiment}")

# Ten en cuenta que este es un ejemplo muy simplificado y que en aplicaciones de procesamiento de lenguaje natural más avanzadas, se explorarán aspectos adicionales del PPL y se utilizarán arquitecturas más complejas.
