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

# 🔢 Prepare Sequences
sequence_length = 100
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
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

st.info("🧠 Training model (this may take a while)...")
model.fit(X, y, epochs=5, batch_size=64)
st.success("✅ Model trained!")

# 🎶 Generate Music
int_to_note = {number: note for note, number in note_to_int.items()}
start = random.randint(0, len(network_input) - 1)
pattern = network_input[start]
generated_notes = []

for note_index in range(300):
    input_seq = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
    prediction = model.predict(input_seq, verbose=0)
    index = np.argmax(prediction)
    result = int_to_note[index]
    generated_notes.append(result)
    pattern.append(index)
    pattern = pattern[1:]

# 💾 Convert to MIDI and Save
def create_midi(prediction_output, filename="generated_classical.mid"):
    output_notes = []
    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            notes_in_chord = [note.Note(int(n)) for n in pattern.split('.')]
            new_chord = chord.Chord(notes_in_chord)
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            output_notes.append(new_note)
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=filename)

create_midi(generated_notes)
st.success("🎵 Music generated and saved as `generated_classical.mid`!")


