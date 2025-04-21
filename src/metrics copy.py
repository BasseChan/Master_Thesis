import numpy as np
from pymcd.mcd import Calculate_MCD
import librosa
from typing import Literal
from tqdm import tqdm
import re


mcd_toolbox = Calculate_MCD(MCD_mode="dtw")


def get_SDR(reference, estimated):
    reference = reference.reshape(reference.size, 1)
    estimated = estimated.reshape(estimated.size, 1)

    eps = np.finfo(reference.dtype).eps

    Sss = (reference**2).sum()
    Snn = ((estimated - reference)**2).sum()

    return 10 * np.log10((eps + Sss) / (eps + Snn))


def get_SI_SDR(reference, estimated):
    reference = reference.reshape(reference.size, 1)
    estimated = estimated.reshape(estimated.size, 1)

    eps = np.finfo(reference.dtype).eps
    a = np.dot(estimated.T, reference) / (np.dot(reference.T, reference) + eps)
    reference_scaled = reference * a

    Sss = (reference_scaled**2).sum()
    Snn = ((estimated - reference_scaled)**2).sum()

    return 10 * np.log10((eps + Sss) / (eps + Snn))


def get_PESQ(reference, estimated, sr):
    from pesq import pesq
    return pesq(sr, reference, estimated, 'wb')


def get_STOI(reference, estimated, sr):
    from pystoi import stoi
    return stoi(reference, estimated, sr, extended=False)


def get_MCD(reference_file, estimated_file):
    return mcd_toolbox.calculate_mcd(reference_file, estimated_file)


def normalize_text(text):
    """ Normalizacja tekstu: małe litery, usunięcie interpunkcji, nadmiarowych spacji """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_WER(ref_text, est_text):
    import jiwer
    return jiwer.wer(ref_text, est_text)


def get_CER(ref_text, est_text):
    import jiwer
    return jiwer.cer(ref_text, est_text)


def get_SIM(ref_text, est_text):
    from difflib import SequenceMatcher
    return SequenceMatcher(None, ref_text, est_text).ratio()
    

def match_size(ref_wav, gen_wav):
    if len(ref_wav) > len(gen_wav):
        gen_wav = np.pad(gen_wav, (0, len(ref_wav) - len(gen_wav)), 'constant')
    elif len(ref_wav) < len(gen_wav):
        gen_wav = gen_wav[:len(ref_wav)]
    return gen_wav


def calculate_metrics(reference_files: list[str], estimated_files: list[str] | dict[list[str]],
                      mixed_files: list[str], sr: int = 16000, texts: list[str] = None,
                      metrics: set[Literal["SDR", "SDRi", "SI-SDR", "SI-SDRi", "PESQ", "STOI", "MCD"]] = None,
                      transcription_model = None) -> dict[str, float] | dict[str, dict[str, float]]:
    """
    Calculate metrics for a list of reference and estimated files.

    Args:
        reference_files (list[str]): List of reference audio file paths.
        estimated_files (list[str] | dict[list[str]]): List of estimated audio file paths or a dictionary with keys as labels and values as lists of file paths.
        mixed_files (list[str]): List of mixed audio file paths.
        sr (int): Sample rate for loading audio files.

    Returns:
        dict[str, float] | dict[str, dict[str, float]]: Dictionary with metrics as keys and lists of metric values as values. If estimated_files is a dictionary, the keys will be the labels.
    """

    if metrics is None:
        metrics = {"SDR", "SDRi", "SI-SDR", "SI-SDRi", "PESQ", "STOI", "MCD", "WER", "CER", "SIM", "WER_ratio", "CER_ratio", "SIM_ratio"}
    else:
        metrics = metrics & {"SDR", "SDRi", "SI-SDR", "SI-SDRi", "PESQ", "STOI", "MCD", "WER", "CER", "SIM", "WER_ratio", "CER_ratio", "SIM_ratio"}

    if isinstance(estimated_files, dict):
        results = {label: {metric: [] for metric in metrics} for label in estimated_files.keys()}
    else:
        results = {metric: [] for metric in metrics}

    # ref_wavs = [librosa.load(ref_file, sr=sr, mono=True)[0] for ref_file in reference_files]

    if "SDRi" in metrics or "SI-SDRi" in metrics:
        if "SDRi" in metrics:
            sdr_mix = []
        if "SI-SDRi" in metrics:
            si_sdr_mix = []
        for ref_file, mix_file in tqdm(list(zip(reference_files, mixed_files))):
            ref = librosa.load(ref_file, sr=sr, mono=True)[0]
            mix = librosa.load(mix_file, sr=sr, mono=True)[0]
            if "SDRi" in metrics:
                sdr_mix.append(get_SDR(ref, mix))
            if "SI-SDRi" in metrics:
                si_sdr_mix.append(get_SI_SDR(ref, mix))

    if {"WER", "CER", "SIM", "WER_ratio", "CER_ratio", "SIM_ratio"} & metrics:
        if texts is None:
            raise ValueError("Texts must be provided for WER, CER, SIM, WER_ratio, CER_ratio, or SIM_ratio metrics.")
        texts = [normalize_text(text) for text in texts]
    
    if "WER_ratio" in metrics or "CER_ratio" in metrics or "SIM_ratio" in metrics:
        if "WER_ratio" in metrics:
            wer_ref = []
        if "CER_ratio" in metrics:
            cer_ref = []
        if "SIM_ratio" in metrics:
            sim_ref = []
        for ref_file, text in tqdm(list(zip(reference_files, texts)), desc='Kontrolne wyliczenie metryk dla referencji'):
            ref = librosa.load(ref_file, sr=sr, mono=True)[0]
            asr_text = normalize_text(transcription_model.transcribe(ref, without_timestamps=True, language='en')['text'].strip())
            if "WER_ratio" in metrics:
                wer_ref.append(get_WER(text, asr_text))
            if "CER_ratio" in metrics:
                cer_ref.append(get_CER(text, asr_text))
            if "SIM_ratio" in metrics:
                sim_ref.append(get_SIM(text, asr_text))
    
    if isinstance(estimated_files, dict):
        for i in tqdm(range(len(reference_files))):
            ref = librosa.load(reference_files[i], sr=sr, mono=True)[0]
            for model in estimated_files.keys():
                est_file = estimated_files[model][i]
                est_wav = librosa.load(est_file, sr=sr, mono=True)[0]
                est_wav = match_size(ref, est_wav)

                if "SDR" in metrics or "SDRi" in metrics:
                    sdr = get_SDR(ref, est_wav)
                    if "SDR" in metrics:
                        results[model]["SDR"].append(sdr)
                    if "SDRi" in metrics:
                        results[model]["SDRi"].append(sdr - sdr_mix[i])
                
                if "SI-SDR" in metrics or "SI-SDRi" in metrics:
                    si_sdr = get_SI_SDR(ref, est_wav)
                    if "SI-SDR" in metrics:
                        results[model]["SI-SDR"].append(si_sdr)
                    if "SI-SDRi" in metrics:
                        results[model]["SI-SDRi"].append(si_sdr - si_sdr_mix[i])

                if "PESQ" in metrics:
                    pesq = get_PESQ(ref, est_wav, sr)
                    results[model]["PESQ"].append(pesq)

                if "STOI" in metrics:
                    stoi = get_STOI(ref, est_wav, sr)
                    results[model]["STOI"].append(stoi)

                if "MCD" in metrics:
                    mcd = get_MCD(reference_files[i], estimated_files[model][i])
                    results[model]["MCD"].append(mcd)

                if {"WER", "CER", "SIM", "WER_ratio", "CER_ratio", "SIM_ratio"} & metrics:
                    asr_text = normalize_text(transcription_model.transcribe(est_wav, without_timestamps=True, language='en')['text'].strip())
                    if "WER" in metrics or "WER_ratio" in metrics:
                        wer = get_WER(texts[i], asr_text)
                        if "WER" in metrics:
                            results[model]["WER"].append(wer)
                        if "WER_ratio" in metrics:
                            results[model]["WER_ratio"].append((1 - wer) / (1 - wer_ref[i] + 1e-10))
                    if "CER" in metrics or "CER_ratio" in metrics:
                        cer = get_CER(texts[i], asr_text)
                        if "CER" in metrics:
                            results[model]["CER"].append(cer)
                        if "CER_ratio" in metrics:
                            results[model]["CER_ratio"].append((1 - cer) / (1 - cer_ref[i] + 1e-10))
                    if "SIM" in metrics or "SIM_ratio" in metrics:
                        sim = get_SIM(texts[i], asr_text)
                        if "SIM" in metrics:
                            results[model]["SIM"].append(sim)
                        if "SIM_ratio" in metrics:
                            results[model]["SIM_ratio"].append(sim / (sim_ref[i] + 1e-10))

                print(f"Model: {model}, File: {i}, SDR: {results[model]['SDR'][-1]}, SDRi: {results[model]['SDRi'][-1]}, SI-SDR: {results[model]['SI-SDR'][-1]}, SI-SDRi: {results[model]['SI-SDRi'][-1]}")
                print(f"Model: {model}, PESQ: {results[model]['PESQ'][-1]}, STOI: {results[model]['STOI'][-1]}, MCD: {results[model]['MCD'][-1]}")
                print(f"WER: {results[model]['WER'][-1]}, CER: {results[model]['CER'][-1]}, SIM: {results[model]['SIM'][-1]}, WER_ratio: {results[model]['WER_ratio'][-1]}, CER_ratio: {results[model]['CER_ratio'][-1]}, SIM_ratio: {results[model]['SIM_ratio'][-1]}")

        for model in estimated_files.keys():
            for metric in metrics:
                results[model][metric] = np.mean(results[model][metric])

    else:
        for i in tqdm(range(len(reference_files))):
            est_file = librosa.load(estimated_files[i], sr=sr, mono=True)[0]
            est_wav = librosa.load(est_file, sr=sr, mono=True)[0]
            est_wav = match_size(ref, est_wav)

            if "SDR" in metrics or "SDRi" in metrics:
                sdr = get_SDR(est_file, est_wav)
                if "SDR" in metrics:
                    results["SDR"].append(sdr)
                if "SDRi" in metrics:
                    results["SDRi"].append(sdr - sdr_mix[i])
            
            if "SI-SDR" in metrics or "SI-SDRi" in metrics:
                si_sdr = get_SI_SDR(est_file, est_wav)
                if "SI-SDR" in metrics:
                    results["SI-SDR"].append(si_sdr)
                if "SI-SDRi" in metrics:
                    results["SI-SDRi"].append(si_sdr - si_sdr_mix[i])

            if "PESQ" in metrics:
                pesq = get_PESQ(est_file, est_wav, sr)
                results["PESQ"].append(pesq)

            if "STOI" in metrics:
                stoi = get_STOI(est_file, est_wav, sr)
                results["STOI"].append(stoi)

            if "MCD" in metrics:
                mcd = get_MCD(reference_files[i], estimated_files[i])
                results["MCD"].append(mcd)

            if {"WER", "CER", "SIM", "WER_ratio", "CER_ratio", "SIM_ratio"} & metrics:
                asr_text = normalize_text(transcription_model.transcribe(est_wav, without_timestamps=True, language='en')['text'].strip())
                if "WER" in metrics or "WER_ratio" in metrics:
                    wer = get_WER(texts[i], asr_text)
                    if "WER" in metrics:
                        results["WER"].append(wer)
                    if "WER_ratio" in metrics:
                        results["WER_ratio"].append((1 - wer) / (1 - wer_ref[i] + 1e-10))
                if "CER" in metrics or "CER_ratio" in metrics:
                    cer = get_CER(texts[i], asr_text)
                    if "CER" in metrics:
                        results["CER"].append(cer)
                    if "CER_ratio" in metrics:
                        results["CER_ratio"].append((1 - cer) / (1 - cer_ref[i] + 1e-10))
                if "SIM" in metrics or "SIM_ratio" in metrics:
                    sim = get_SIM(texts[i], asr_text)
                    if "SIM" in metrics:
                        results["SIM"].append(sim)
                    if "SIM_ratio" in metrics:
                        results["SIM_ratio"].append(sim / (sim_ref[i] + 1e-10))
        
        for metric in metrics:
            results[metric] = np.mean(results[metric])

    return results