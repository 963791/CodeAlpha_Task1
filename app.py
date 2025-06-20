import streamlit as st
import numpy as np
import json
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.utils import to_categorical  
from music21 import note, chord, stream
import random

st.title("🎼 AI Classical Music Generator")

# 📥 Load Notes from Preprocessed JSON
with open("classical-music-midi-metadata.json", "r") as f:
    notes = json.load(f)

st.success(f"✅ Loaded {len(notes)} notes from JSON.")

# 🚨 Check if enough notes exist
sequence_length = 100
if len(notes) < sequence_length:
    st.error(f"❌ Not enough notes to create sequences. At least {sequence_length} notes required.")
    st.stop()

# 🔢 Prepare Sequences
pitch_names = sorted(set(notes))
note_to_int = {note: number for number, note in enumerate(pitch_names)}

network_input = []
network_output = []

for i in range(len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]
    network_input.append([note_to_int[n] for n in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)
n_vocab = len(pitch_names)

X = np.reshape(network_input, (n_patterns, sequence_length, 1)) / float(n_vocab)
y = to_categorical(network_output)

# 🧠 Build the LSTM Model
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='_
