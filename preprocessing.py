## Compute the ppg of basic5000 .wav data with phonetic transcripts in basic5000.yaml

import yaml

import torchaudio
import torch 

import IPython
import matplotlib.pyplot as plt

import torchaudio.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

SPEECH_FILE = "data/basic5000/wav/BASIC5000_0001.wav"
waveform, _ = torchaudio.load(SPEECH_FILE)

TRANSCRIPTS_FILE = "data/basic5000.yaml"

with open(TRANSCRIPTS_FILE, "r") as f:
    TRANSCRIPTS = yaml.safe_load(f)
    TRANSCRIPTS = TRANSCRIPTS["BASIC5000_0001"]["phone_level3"].split("-")

print("Phonetic Trancript is", TRANSCRIPTS)
print("Waveform shape is", waveform.shape)

bundle = torchaudio.pipelines.MMS_FA

model = bundle.get_model(with_star=False).to(device)
with torch.inference_mode():
    emission, _ = model(waveform.to(device))

def plot_emission(emission, filename):
    fig, ax = plt.subplots()
    ax.imshow(emission.cpu().T)
    ax.set_title("Frame-wise class probabilities")
    ax.set_xlabel("Time")
    ax.set_ylabel("Labels")
    fig.tight_layout()
    plt.savefig(filename)

plot_emission(emission, "emission_plot.png")

LABELS = bundle.get_labels(star=None)
DICTIONARY = bundle.get_dict(star=None)

print("Labels are", LABELS)

tokenized_transcript = [DICTIONARY[c] for word in TRANSCRIPTS for c in word]

#computing alignments

def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores


aligned_tokens, alignment_scores = align(emission, tokenized_transcript)

token_spans = F.merge_tokens(aligned_tokens, alignment_scores)

def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret

word_spans = unflatten(token_spans, [len(word) for word in TRANSCRIPTS])

def _score(spans):
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)


def preview_word(waveform, spans, num_frames, transcript, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / num_frames
    x0 = int(ratio * spans[0].start)
    x1 = int(ratio * spans[-1].end)
    print(f"{transcript} ({_score(spans):.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec")
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=sample_rate)


num_frames = emission.size(1)

def plot_alignments(waveform, token_spans, emission, transcript, filename, sample_rate=bundle.sample_rate):
    ratio = waveform.size(1) / emission.size(1) / sample_rate

    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(emission[0].detach().cpu().T, aspect="auto")
    axes[0].set_title("Emission")
    axes[0].set_xticks([])

    axes[1].specgram(waveform[0], Fs=sample_rate)
    for t_spans, chars in zip(token_spans, transcript):
        t0, t1 = t_spans[0].start + 0.1, t_spans[-1].end - 0.1
        axes[0].axvspan(t0 - 0.5, t1 - 0.5, facecolor="None", hatch="/", edgecolor="white")
        axes[1].axvspan(ratio * t0, ratio * t1, facecolor="None", hatch="/", edgecolor="white")
        axes[1].annotate(f"{_score(t_spans):.2f}", (ratio * t0, sample_rate * 0.51), annotation_clip=False)

        for span, char in zip(t_spans, chars):
            t0 = span.start * ratio
            axes[1].annotate(char, (t0, sample_rate * 0.55), annotation_clip=False)

    axes[1].set_xlabel("time [second]")
    axes[1].set_xlim([0, None])
    fig.tight_layout()
    plt.savefig(filename)

plot_alignments(waveform, word_spans, emission, TRANSCRIPTS, 'alignment_plot.png')