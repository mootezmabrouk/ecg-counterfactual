import numpy as np
import neurokit2 as nk


def safe_int(x):
    """Boundaries can be NaN if detection failed for that beat — handle it."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return int(x)


def extract_segment(signal, onset, offset):
    onset, offset = safe_int(onset), safe_int(offset)
    if onset is None or offset is None or onset >= offset:
        return None
    return signal[onset:offset]


def segment_all_beats(record, sampling_rate=250, reference_lead=1):
    """
    record: shape (2500, 12) — one patient, all leads, raw
    Returns nested dict: {beat_idx: {lead_idx: {segment_name: array or None}}}
    """
    # --- detect R-peaks + delineate ONCE, on the reference lead only ---
    reference_signal = record[:, reference_lead]
    cleaned_ref = nk.ecg_clean(reference_signal, sampling_rate=sampling_rate)
    _, rpeaks_info = nk.ecg_peaks(cleaned_ref, sampling_rate=sampling_rate)
    rpeaks = rpeaks_info["ECG_R_Peaks"]
    _, waves_info = nk.ecg_delineate(cleaned_ref, rpeaks_info, sampling_rate=sampling_rate, method="dwt")

    results = {}  # fix 3: dict, not set

    for i in range(len(rpeaks)):  # fix 6: include last beat
        beat_result = {}

        for lead in range(12):
            # fix 2: use the record passed into the function, no hardcoded reload
            # fix 1: no re-detection here — just clean this lead's raw signal for extraction
            lead_signal = record[:, lead]
            cleaned_lead = nk.ecg_clean(lead_signal, sampling_rate=sampling_rate)

            # fix 5: use i (the real loop variable), not the undefined beat_idx
            p_wave = extract_segment(
                cleaned_lead, waves_info["ECG_P_Onsets"][i], waves_info["ECG_P_Offsets"][i]
            )
            qrs = extract_segment(
                cleaned_lead, waves_info["ECG_R_Onsets"][i], waves_info["ECG_R_Offsets"][i]
            )
            st_segment = extract_segment(
                cleaned_lead, waves_info["ECG_R_Offsets"][i], waves_info["ECG_T_Onsets"][i]
            )
            t_wave = extract_segment(
                cleaned_lead, waves_info["ECG_T_Onsets"][i], waves_info["ECG_T_Offsets"][i]
            )

            # fix 3 & 4: plain dict with integer keys, not set + undefined string concat
            beat_result[lead] = {
                "p_wave": p_wave,
                "qrs": qrs,
                "st_segment": st_segment,
                "t_wave": t_wave,
            }

        results[i] = beat_result

    return results
waveforms = np.load("data/EchoNext_val_waveforms.npy")
record = waveforms[0, 0, :, :]

results = segment_all_beats(record)
print("number of beats segmented:", len(results))
print("beat 0, lead 1, QRS segment:", results[0][1]["qrs"])