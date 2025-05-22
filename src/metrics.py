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


def get_MSS(reference, estimated):
    S_ref = np.abs(librosa.stft(reference, n_fft=1024, hop_length=512))
    S_gen = np.abs(librosa.stft(estimated, n_fft=1024, hop_length=512))

    S_max = np.maximum(S_ref, S_gen)
    return (sum(sum(S_ref)) + sum(sum(S_gen))) / sum(sum(S_max)) - 1


def get_SI_MSS(reference, estimated):
    eps = np.finfo(reference.dtype).eps
    a = np.dot(estimated.T, reference) / (np.dot(reference.T, reference) + eps)
    reference = reference * a

    return get_MSS(reference, estimated)


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


def calculate_single_file_metrics(reference_file: str, estimated_file: str | dict[str],
                      mixed_file: str, sr: int = 16000, text: str = None,
                      metrics: set[Literal["SDR", "SDRi", "SI-SDR", "SI-SDRi", "PESQ", "STOI", "MCD", "MSS", "SI-MSS"]] = None,
                      transcription_model = None) -> dict[str, float] | dict[str, dict[str, float]]:
    """
    Calculate metrics for one reference and estimated file.

    Args:
        reference_file (str): Path to the reference audio file.
        estimated_file (str | dict[str]): Path to the estimated audio file or a dictionary with keys as labels and values as paths.
        mixed_file (str): Path to the mixed audio file.
        sr (int): Sample rate for loading audio files.
        metrics (set[Literal["SDR", "SDRi", "SI-SDR", "SI-SDRi", "PESQ", "STOI", "MCD", "MSS", "SI-MSS"]], optional): Set of metrics to calculate. Defaults to None, which calculates all metrics.
        transcription_model: Transcription model for WER, CER, SIM metrics.

    Returns:
        dict[str, float] | dict[str, dict[str, float]]: Dictionary with metrics as keys and lists of metric values as values. If estimated_files is a dictionary, the keys will be the labels.
    """

    if metrics is None:
        metrics = {"SDR", "SDRi", "SI-SDR", "SI-SDRi", "PESQ", "STOI", "MCD", "WER", "CER", "SIM", "MSS", "SI-MSS"}
    else:
        metrics = metrics & {"SDR", "SDRi", "SI-SDR", "SI-SDRi", "PESQ", "STOI", "MCD", "WER", "CER", "SIM", "MSS", "SI-MSS"}

    ref = librosa.load(reference_file, sr=sr, mono=True)[0]
    if sr != 16000: ref = librosa.resample(ref, orig_sr=sr, target_sr=16000)

    if "SDRi" in metrics or "SI-SDRi" in metrics:
        mix = librosa.load(mixed_file, sr=sr, mono=True)[0]

        if sr != 16000: mix = librosa.resample(mix, orig_sr=sr, target_sr=16000)

        if "SDRi" in metrics: sdr_mix = get_SDR(ref, mix)
        if "SI-SDRi" in metrics: si_sdr_mix = get_SI_SDR(ref, mix)

    if "WER" in metrics or "CER" in metrics or "SIM" in metrics:
        if transcription_model is None:
            raise ValueError("Transcription model is required for WER, CER, and SIM metrics.")
        text = normalize_text(text)

    def _get_metrics(ref_file, est_file, ref, est, metrics):
        results = {}
        if "SDR" in metrics or "SDRi" in metrics:
            sdr = get_SDR(ref, est)
            if "SDR" in metrics:
                results["SDR"] = sdr
            if "SDRi" in metrics:
                results["SDRi"] = sdr - sdr_mix

        if "SI-SDR" in metrics or "SI-SDRi" in metrics:
            si_sdr = get_SI_SDR(ref, est)
            if "SI-SDR" in metrics:
                results["SI-SDR"] = si_sdr
            if "SI-SDRi" in metrics:
                results["SI-SDRi"] = si_sdr - si_sdr_mix

        if "PESQ" in metrics:
            pesq = get_PESQ(ref, est, 16000)
            results["PESQ"] = pesq

        if "STOI" in metrics:
            stoi = get_STOI(ref, est, 16000)
            results["STOI"] = stoi

        if "MCD" in metrics:
            mcd = get_MCD(ref_file, est_file)
            results["MCD"] = mcd

        if {"WER", "CER", "SIM"} & metrics:
            asr_text = normalize_text(transcription_model.transcribe(est, without_timestamps=True, language='en')['text'].strip())
            if "WER" in metrics:
                wer = get_WER(text, asr_text)
                results["WER"] = wer
            if "CER" in metrics:
                cer = get_CER(text, asr_text)
                results["CER"] = cer
            if "SIM" in metrics:
                sim = get_SIM(text, asr_text)
                results["SIM"] = sim
        
        if "MSS" in metrics:
            mss = get_MSS(ref, est)
            results["MSS"] = mss

        if "SI-MSS" in metrics:
            si_mss = get_SI_MSS(ref, est)
            results["SI-MSS"] = si_mss

        return results

    if isinstance(estimated_file, dict):
        results = {}
        for model, est_file in estimated_file.items():
            est = librosa.load(est_file, sr=sr, mono=True)[0]
            if sr != 16000: est = librosa.resample(est, orig_sr=sr, target_sr=16000)
            est = match_size(ref, est)

            results[model] = _get_metrics(reference_file, est_file, ref, est, metrics)
    else:
        est = librosa.load(estimated_file, sr=sr, mono=True)[0]
        if sr != 16000: est = librosa.resample(est, orig_sr=sr, target_sr=16000)
        est = match_size(ref, est)

        results = _get_metrics(reference_file, estimated_file, ref, est, metrics)

    return results


def calculate_metrics(reference_files: list[str], estimated_files: list[str] | list[dict[str]],
                      mixed_files: list[str], sr: int = 16000, texts: list[str] = None,
                      metrics: set[Literal["SDR", "SDRi", "SI-SDR", "SI-SDRi", "PESQ", "STOI", "MCD", "MSS", "SI-MSS"]] = None,
                      transcription_model = None) -> list[dict[str, float]] | list[dict[str, dict[str, float]]]:
    """
    Calculate metrics for a list of reference and estimated files.

    Args:
        reference_files (list[str]): List of reference audio file paths.
        estimated_files (list[str] | list[dict[str]]): List of estimated audio file paths or a dictionary with keys as labels and values as lists of file paths.
        mixed_files (list[str]): List of mixed audio file paths.
        sr (int): Sample rate for loading audio files.

    Returns:
        dict[str, float] | dict[str, dict[str, float]]: Dictionary with metrics as keys and lists of metric values as values. If estimated_files is a dictionary, the keys will be the labels.
    """

    if metrics is None:
        metrics = {"SDR", "SDRi", "SI-SDR", "SI-SDRi", "PESQ", "STOI", "MCD", "WER", "CER", "SIM", "MSS", "SI-MSS"}
    else:
        metrics = metrics & {"SDR", "SDRi", "SI-SDR", "SI-SDRi", "PESQ", "STOI", "MCD", "WER", "CER", "SIM", "MSS", "SI-MSS"}

    results = [calculate_single_file_metrics(ref_file, est_file, mix_file, sr, text, metrics, transcription_model)
               for ref_file, est_file, mix_file, text in tqdm(list(zip(reference_files, estimated_files, mixed_files, texts)))]
    
    return results