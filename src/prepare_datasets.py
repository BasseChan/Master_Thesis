import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
from scipy.io.wavfile import write
import random
import json
from collections import deque
import shutil


def prepare_2_mix_dataset(save_location, min_length, max_length, min_overlap, max_overlap, n=1000, sr=16000, seed=42):
    np.random.seed(seed)
    df = pd.read_csv('data/metadata.csv')
    df = df[(df['duration'] >= min_length) & (df['duration'] <= max_length)]

    os.makedirs(save_location, exist_ok=False)
    os.makedirs(os.path.join(save_location, 'mixed'), exist_ok=True)
    os.makedirs(os.path.join(save_location, 'speaker1'), exist_ok=True)
    os.makedirs(os.path.join(save_location, 'speaker2'), exist_ok=True)

    texts = []
    for i in tqdm(range(n)):
        sample = df.sample(2, random_state=seed)
        while sample.iloc[0]['speaker'] == sample.iloc[1]['speaker']:
            sample = df.sample(2)
        # wczytaj pliki o nazwach z sample
        wav1 = librosa.load(f'data/voices/{sample.iloc[0]["file"]}.wav', sr=sr)[0]
        wav2 = librosa.load(f'data/voices/{sample.iloc[1]["file"]}.wav', sr=sr)[0]
        wav1 = wav1 / np.max(np.abs(wav1))
        wav2 = wav2 / np.max(np.abs(wav2))
        # wylosuj długość overlapu
        overlap_length = min(int(np.random.uniform(min_overlap, max_overlap) * sr), len(wav1), len(wav2))
        # nałuż próbki z overlapem zadanym przez overlap_length
        overlap_sample = np.concatenate((wav1[:-overlap_length], wav1[-overlap_length:] + wav2[:overlap_length], wav2[overlap_length:]))
        overlap_sample = overlap_sample / np.max(np.abs(overlap_sample))
        # dołącz ciszę po wav1
        speaker1 = np.concatenate((wav1, np.zeros(overlap_sample.shape[0] - wav1.shape[0], dtype=np.float32)))
        speaker2 = np.concatenate((np.zeros(overlap_sample.shape[0] - wav2.shape[0], dtype=np.float32), wav2))

        # zapisz do pliku
        write(os.path.join(save_location, f'mixed/{i}.wav'), sr, overlap_sample)
        write(os.path.join(save_location, f'speaker1/{i}.wav'), sr, speaker1)
        write(os.path.join(save_location, f'speaker2/{i}.wav'), sr, speaker2)

        texts.append([sample.iloc[0]['text'], sample.iloc[1]['text']])

    texts_df = pd.DataFrame(texts, columns=['text1', 'text2'])
    texts_df.to_csv(os.path.join(save_location, f'texts.csv'), index=False)


def smart_shuffle(data):
    to_add = data[:]
    random.shuffle(to_add)
    to_add = deque(to_add)
    result = []
    last_match = 0
    while to_add:
        if last_match > len(to_add):
            for item in to_add:
                result.insert(random.randint(0, len(data)), item)
            break
        elem = to_add.popleft()
        if not result:
            result.append(elem)
            continue
        possible_idx = []
        if elem['speaker'] != result[0]['speaker']:
            possible_idx.append(0)
        if elem['speaker'] != result[-1]['speaker'] and len(result) > 1:
            possible_idx.append(len(result))
        for i in range(1, len(result) - 1):
            if elem['speaker'] != result[i]['speaker'] and elem['speaker'] != result[i - 1]['speaker']:
                possible_idx.append(i)
        if possible_idx:
            idx = random.choice(possible_idx)
            result.insert(idx, elem)
            last_match = 0
        else:
            to_add.append(elem)
            last_match += 1
    return result


def prepare_dialogue_sample(df, overlap_prob=0.7, min_speakers=2, max_speakers=5,
                                              min_fragments_per_speaker=1, max_fragments_per_speaker=3,
                                              min_speaker_time=8, min_overlap=1, max_overlap=8,
                                              min_space=0.5, max_space=1.5,
                                              allow_repeat_speakers=True, sr=16000):
    # Wybór losowych mówców
    speakers = random.sample(df['speaker'].unique().tolist(), random.randint(min_speakers, max_speakers))

    # Wybór losowych nagrań dla każdego mówcy
    samples = []
    for speaker in speakers:
        files = set()
        speaker_time = 0
        speaker_fragments_number = random.randint(min_fragments_per_speaker, max_fragments_per_speaker)
        speaker_fragments = df[df['speaker'] == speaker].sample(speaker_fragments_number)
        for i in range(speaker_fragments_number):
            speaker_time += speaker_fragments.iloc[i]['duration']
            samples.append(speaker_fragments.iloc[i].to_dict())
            files.add(speaker_fragments.iloc[i]['file'])
        while speaker_time < min_speaker_time:
            speaker_fragment = df[df['speaker'] == speaker].sample(1)
            if speaker_fragment.iloc[0]['file'] not in files:
                speaker_time += speaker_fragment.iloc[0]['duration']
                samples.append(speaker_fragment.iloc[0].to_dict())
                files.add(speaker_fragment.iloc[0]['file'])
    
    if allow_repeat_speakers:
        random.shuffle(samples)
    else:
        samples = smart_shuffle(samples)

    connected_wav = librosa.load(f'data/voices/{samples[0]["file"]}.wav', sr=sr)[0]
    overlap = []
    data = [[samples[0]["file"], samples[0]['speaker'], samples[0]['text'], 0, connected_wav.shape[0] / sr]]
    prev_speaker = samples[0]['speaker']
    prev_fragment_end = 0
    for i in range(1, len(samples)):
        wav = librosa.load(f'data/voices/{samples[i]["file"]}.wav', sr=sr)[0]

        # losuj odstęp od poprzedniego fragmentu
        if samples[i]['speaker'] == prev_speaker:
            connection_time = random.randint(int(min_space * sr), int(max_space * sr))
        else:
            if random.uniform(0, 1) > overlap_prob:
                connection_time = random.randint(int(min_space * sr), int(max_space * sr))
            else:
                max_delay = min(connected_wav.shape[0] - prev_fragment_end - int(max_space * sr), int(max_overlap * sr))
                if max_delay < int(min_overlap * sr):
                    connection_time = random.randint(int(min_space * sr), int(max_space * sr))
                else:
                    connection_time = -random.randint(int(min_overlap * sr), max_delay)

        # połącz fragment
        if connection_time > 0: # brak overlapa
            prev_fragment_end = connected_wav.shape[0]
            connected_wav = np.concatenate((connected_wav, np.zeros(connection_time, dtype=np.float32), wav))
            prev_speaker = samples[i]['speaker']
            data.append([samples[i]["file"], samples[i]['speaker'], samples[i]['text'], (connected_wav.shape[0] - wav.shape[0]) / sr, connected_wav.shape[0] / sr])
        else: # overlap
            connection_time = -connection_time
            if wav.shape[0] <= connection_time:
                connected_wav[-connection_time : -connection_time + wav.shape[0]] += wav
                end = connected_wav.shape[0] - connection_time + wav.shape[0]
                overlap.append([(connected_wav.shape[0] - connection_time) / sr, end / sr])
                data.append([samples[i]["file"], samples[i]['speaker'], samples[i]['text'], (connected_wav.shape[0] - connection_time) / sr, end / sr])
                # if end > prev_fragment_end:
                prev_fragment_end = end
                # organized_fragments.append([samples[i]['file'], samples[i]['text'], samples[i]['speaker'], end - wav.shape[0], end])
            else:
                prev_fragment_end = connected_wav.shape[0]
                connected_wav[-connection_time:] += wav[:connection_time]
                overlap.append([(connected_wav.shape[0] - connection_time) / sr, connected_wav.shape[0] / sr])
                connected_wav = np.concatenate((connected_wav, wav[connection_time:]))
                prev_speaker = samples[i]['speaker']
                data.append([samples[i]["file"], samples[i]['speaker'], samples[i]['text'], (connected_wav.shape[0] - wav.shape[0]) / sr, connected_wav.shape[0] / sr])
                # organized_fragments.append([samples[i]['file'], samples[i]['text'], samples[i]['speaker'], connected_wav.shape[0] - wav.shape[0], connected_wav.shape[0]])
    connected_wav = connected_wav / np.max(np.abs(connected_wav))
    return connected_wav, data, overlap


def prepare_dialogue_dataset(n, save_dir, overlap_prob=0.7, min_speakers=2, max_speakers=5,
                                              min_fragments_per_speaker=1, max_fragments_per_speaker=3,
                                              min_speaker_time=8, min_overlap=1, max_overlap=8,
                                              min_space=0.5, max_space=1.5,
                                              allow_repeat_speakers=True, sr=16000, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    df = pd.read_csv('data/metadata.csv')
    df_filterd = df[(df['duration'] >= 1) & (df['duration'] <= 10)]
    speakers = df_filterd.groupby('speaker')['duration'].sum()
    speakers = speakers[speakers >= 30]
    df_filterd = df_filterd[df_filterd['speaker'].isin(speakers.index)]

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=False)

    total_time = 0
    overlap_time = 0
    overlaps = []
    fragments = []
    for i in tqdm(range(n)):
        connected_wav, data, overlap = prepare_dialogue_sample(df_filterd, overlap_prob, min_speakers, max_speakers,
                                                        min_fragments_per_speaker, max_fragments_per_speaker,
                                                        min_speaker_time, min_overlap, max_overlap,
                                                        min_space, max_space,
                                                        allow_repeat_speakers, sr)
        total_time += connected_wav.shape[0] / sr
        for j in range(len(overlap)):
            overlaps.append([i, *overlap[j]])
            overlap_time += overlap[j][-1] - overlap[j][-2]
        for j in range(len(data)):
            fragments.append([i, *data[j]])
        write(f'{save_dir}/{i}.wav', sr, connected_wav)
    overlaps = pd.DataFrame(overlaps, columns=['sample_id', 'start', 'end'])
    overlaps.to_csv(f'{save_dir}/overlaps.csv', index=False)
    fragments = pd.DataFrame(fragments, columns=['sample_id', 'file', 'speaker', 'text', 'start', 'end'])
    fragments.to_csv(f'{save_dir}/fragments.csv', index=False)
    print('Overlap time:', overlap_time / total_time * 100, '%')