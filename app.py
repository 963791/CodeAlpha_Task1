import os
import numpy as np
import streamlit as st
from music21 import converter, instrument, note, chord, stream as m21stream
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.utils import to_categorical
import random

# Page settings
st.set_page_config(page_title="AI Music Generator", layout="wide")
st.title("🎼 LSTM-Based Classical Music Generator")

# 📥 MIDI Note Extraction
def extract_notes_from_midi(midi_dir):
    notes = []
    for file in os.listdir(midi_dir):
        if file.endswith('.mid'):
            try:
                midi = converter.parse(os.path.join(midi_dir, file))
                parts = instrument.partitionByInstrument(midi)
                elements = parts.parts[0].recurse() if parts else midi.flat.notes
                for element in elements:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n) for n in element.normalOrder))
            except Exception as e:
                st.warning(f"Error parsing {file}: {e}")
    return notes

# 🧠 LSTM Model
def build_model(input_shape, n_vocab):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

# 🎶 Generate MIDI File
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
    midi_stream = m21stream.Stream(output_notes)
    midi_stream.write('midi', fp=filename)

# 📂 Upload and Training
uploaded = st.file_uploader("📁 Upload MIDI files (zipped folder)", type=["zip"])
if uploaded:
    with open("midi.zip", "wb") as f:
        f.write(uploaded.getbuffer())
    os.system("unzip -o midi.zip -d ./midi_data")

    st.success("✅ MIDI files uploaded and extracted.")
    notes = extract_notes_from_midi("./midi_data")

    if len(notes) > 100:
        st.info(f"🎵 Total notes extracted: {len(notes)}")
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

        n_vocab = len(pitch_names)
        X = np.reshape(network_input, (len(network_input), sequence_length, 1)) / float(n_vocab)
        y = to_categorical(network_output)

        model = build_model((X.shape[1], 1), n_vocab)
        with st.spinner("🎹 Training model... (few minutes)"):
            model.fit(X, y, epochs=20, batch_size=64, verbose=0)
        st.success("✅ Model trained.")

        st.subheader("🎼 Generate Music")
        if st.button("Generate"):
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

            create_midi(generated_notes)
            with open("generated_classical.mid", "rb") as f:
                st.download_button("🎧 Download Generated MIDI", f, file_name="generated_classical.mid")


