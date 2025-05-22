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
from scipy.spatial.distance import cosine


def prepare_2_mix_dataset(save_location, embeddings, min_length, max_length, min_overlap, max_overlap, n=1000, sr=16000, seed=42, dataset=''):
    if dataset:
        dataset = f'_{dataset}'
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    df_full = pd.read_csv(f'data/metadata{dataset}.csv')
    df = df_full[(df_full['duration'] >= min_length) & (df_full['duration'] <= max_length)].copy()
    df = df.reset_index(drop=False)

    os.makedirs(save_location, exist_ok=False)
    os.makedirs(os.path.join(save_location, 'mixed'), exist_ok=True)
    os.makedirs(os.path.join(save_location, 'speaker1'), exist_ok=True)
    os.makedirs(os.path.join(save_location, 'speaker2'), exist_ok=True)

    data = []
    for i in tqdm(range(n)):
        valid_pair = False
        while not valid_pair:
            indices = rng.choice(len(df), size=2, replace=False)
            s1, s2 = df.iloc[indices[0]], df.iloc[indices[1]]
            valid_pair = s1['speaker'] != s2['speaker']

        wav1 = librosa.load(f'data/voices/{s1["file"]}.wav', sr=sr)[0]
        wav2 = librosa.load(f'data/voices/{s2["file"]}.wav', sr=sr)[0]
        wav1 = wav1 / np.max(np.abs(wav1))
        wav2 = wav2 / np.max(np.abs(wav2))

        overlap_length = min(int(rng.uniform(min_overlap, max_overlap) * sr), len(wav1), len(wav2))
        overlap_sample = np.concatenate((wav1[:-overlap_length], wav1[-overlap_length:] + wav2[:overlap_length], wav2[overlap_length:]))
        overlap_sample = overlap_sample / np.max(np.abs(overlap_sample))

        speaker1 = np.concatenate((wav1, np.zeros(overlap_sample.shape[0] - wav1.shape[0], dtype=np.float32)))
        speaker2 = np.concatenate((np.zeros(overlap_sample.shape[0] - wav2.shape[0], dtype=np.float32), wav2))

        write(os.path.join(save_location, f'mixed/{i}.wav'), sr, overlap_sample)
        write(os.path.join(save_location, f'speaker1/{i}.wav'), sr, speaker1)
        write(os.path.join(save_location, f'speaker2/{i}.wav'), sr, speaker2)

        emb1 = embeddings[s1['index']]
        emb2 = embeddings[s2['index']]
        cos_sim = 1 - cosine(emb1, emb2)

        data.append([
            s1['text'], s2['text'],
            s1['speaker'], s2['speaker'],
            s1['duration'], s2['duration'],
            overlap_sample.shape[0] / sr,
            cos_sim
        ])

    texts_df = pd.DataFrame(data, columns=[
        'text1', 'text2',
        'speaker1', 'speaker2',
        'duration1', 'duration2',
        'duration',
        'cosine_similarity'
    ])
    texts_df.to_csv(os.path.join(save_location, 'texts.csv'), index=False)


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
                                              allow_repeat_speakers=True, sr=16000,
                                              save_format='time'):
    speakers = random.sample(df['speaker'].unique().tolist(), random.randint(min_speakers, max_speakers))

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
    data = [[samples[0]["file"], samples[0]['speaker'], samples[0]['text'], 0, connected_wav.shape[0] / sr]] if save_format == 'time' \
        else [[samples[0]["file"], samples[0]['speaker'], samples[0]['text'], 0, connected_wav.shape[0]]]
    prev_speaker = samples[0]['speaker']
    prev_fragment_end = 0
    for i in range(1, len(samples)):
        wav = librosa.load(f'data/voices/{samples[i]["file"]}.wav', sr=sr)[0]

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

        if connection_time > 0: # brak overlapa
            prev_fragment_end = connected_wav.shape[0]
            connected_wav = np.concatenate((connected_wav, np.zeros(connection_time, dtype=np.float32), wav))
            prev_speaker = samples[i]['speaker']
            data.append([samples[i]["file"], samples[i]['speaker'], samples[i]['text'], (connected_wav.shape[0] - wav.shape[0]) / sr, connected_wav.shape[0] / sr] \
                        if save_format == 'time' \
                        else [samples[i]["file"], samples[i]['speaker'], samples[i]['text'], connected_wav.shape[0] - wav.shape[0], connected_wav.shape[0]])
        else: # overlap
            connection_time = -connection_time
            if wav.shape[0] < connection_time:
                connected_wav[-connection_time : -connection_time + wav.shape[0]] += wav
                end = connected_wav.shape[0] - connection_time + wav.shape[0]
                overlap.append([(connected_wav.shape[0] - connection_time) / sr, end / sr] \
                        if save_format == 'time' \
                        else [connected_wav.shape[0] - connection_time, end])
                data.append([samples[i]["file"], samples[i]['speaker'], samples[i]['text'], (connected_wav.shape[0] - connection_time) / sr, end / sr] \
                        if save_format == 'time' \
                        else [samples[i]["file"], samples[i]['speaker'], samples[i]['text'], connected_wav.shape[0] - connection_time, end])
                prev_fragment_end = end
            else:
                prev_fragment_end = connected_wav.shape[0]
                connected_wav[-connection_time:] += wav[:connection_time]
                overlap.append([(connected_wav.shape[0] - connection_time) / sr, connected_wav.shape[0] / sr] \
                        if save_format == 'time' \
                        else [connected_wav.shape[0] - connection_time, connected_wav.shape[0]])
                connected_wav = np.concatenate((connected_wav, wav[connection_time:]))
                prev_speaker = samples[i]['speaker']
                data.append([samples[i]["file"], samples[i]['speaker'], samples[i]['text'], (connected_wav.shape[0] - wav.shape[0]) / sr, connected_wav.shape[0] / sr] \
                        if save_format == 'time' \
                        else [samples[i]["file"], samples[i]['speaker'], samples[i]['text'], connected_wav.shape[0] - wav.shape[0], connected_wav.shape[0]])
    connected_wav = connected_wav / np.max(np.abs(connected_wav))
    return connected_wav, data, overlap


def prepare_dialogue_dataset(n, save_dir, metadata_path, overlap_prob=0.7, min_speakers=2, max_speakers=5,
                                              min_fragments_per_speaker=1, max_fragments_per_speaker=3,
                                              min_speaker_time=8, min_overlap=1, max_overlap=8,
                                              min_space=0.5, max_space=1.5,
                                              allow_repeat_speakers=True, sr=16000, seed=42,
                                              min_fragment_duration=1, max_fragment_duration=10, min_speaker_reception_time=30,
                                              save_format='time'):
    np.random.seed(seed)
    random.seed(seed)

    df = pd.read_csv(metadata_path)
    df_filterd = df[(df['duration'] >= min_fragment_duration) & (df['duration'] <= max_fragment_duration)]
    speakers = df_filterd.groupby('speaker')['duration'].sum()
    speakers = speakers[speakers >= min_speaker_reception_time]
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
                                                        allow_repeat_speakers, sr, save_format)
        total_time += connected_wav.shape[0]
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
    print('Średnia długość nagrań:', total_time/n/sr)