# Generates the comparison figure for §4.9 Frequency-Biased Sampling.
#
# Bar chart that puts three sampling distributions side by side over the
# actual corpus's most frequent words:
#   - uniform                P(w) = 1 / V
#   - pure frequency         P(w) = count(w) / total
#   - frequency^0.75         P(w) = count(w)^0.75 / sum(count^0.75)
#
# Only the top-N words are shown (otherwise the chart is unreadable on a
# 459-word vocabulary). The three curves are drawn together so the reader
# can see how the ^0.75 smoothing dampens the runaway dominance of
# `the` and `a` while still preferring them over rare words.
#
# Output: output/sketches/sampling-distributions.png
# Style: brand-primary blue (#6471E9) for ^0.75 (the choice we make in
# §4.9), brand-secondary purple (#766C82) for pure frequency (the strong
# alternative), and a muted grey for uniform (the §4.8 default).

import os

import matplotlib.pyplot as plt
import numpy as np

from text_preprocessor import TextPreprocessor


PRIMARY   = "#6471E9"
SECONDARY = "#766C82"
GREY      = "#bbbbbb"
TEXT      = "#1d1f28"
BORDER    = "#dddddd"

OUTPUT_DIR = "output/sketches"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_N    = 12     # Number of most-frequent words to show on the chart
EXPONENT = 0.75   # word2vec smoothing constant


# Pull the corpus and counts
with open("corpus.txt") as f:
    text = f.read()

preprocessor = TextPreprocessor()
_, word_to_id, id_to_word, _, word_counts = preprocessor.preprocess(text)
vocab_size = len(word_to_id)

# Sort words by raw count, take the top-N for the chart
ranked = sorted(word_counts.items(), key=lambda kv: -kv[1])
top_ids   = [wid for wid, _ in ranked[:TOP_N]]
top_words = [id_to_word[wid] for wid in top_ids]
top_counts = [word_counts[wid] for wid in top_ids]

# Probabilities under the three distributions
total_count = sum(word_counts.values())
total_smoothed = sum(c ** EXPONENT for c in word_counts.values())

p_uniform   = [1.0 / vocab_size for _ in top_ids]
p_frequency = [word_counts[wid] / total_count for wid in top_ids]
p_smoothed  = [(word_counts[wid] ** EXPONENT) / total_smoothed for wid in top_ids]


# Plot — three bars per word, grouped
fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_alpha(0)

x = np.arange(len(top_words))
bar_w = 0.27

ax.bar(x - bar_w, p_uniform,   width=bar_w,
       color=GREY,      label="uniform  (P = 1/V)")
ax.bar(x,         p_frequency, width=bar_w,
       color=SECONDARY, label="pure frequency  (count / total)")
ax.bar(x + bar_w, p_smoothed,  width=bar_w,
       color=PRIMARY,
       label=f"frequency$^{{{EXPONENT}}}$  (the §4.9 choice)")

ax.set_xticks(x)
ax.set_xticklabels(top_words, rotation=30, ha="right", color=TEXT, fontsize=11)
ax.set_ylabel("Sampling probability", color=TEXT, fontsize=11)
ax.set_title(
    f"Negative-sampling probability on the top {TOP_N} most frequent words "
    f"(vocab size {vocab_size})",
    color=TEXT, fontsize=12, pad=10,
)
ax.tick_params(colors=TEXT)
ax.legend(frameon=False, loc="upper right")

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(BORDER)

ax.grid(True, axis="y", linestyle=":", color=BORDER, linewidth=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "sampling-distributions.png")
plt.savefig(out, dpi=180, transparent=True, bbox_inches="tight")
plt.close()
print(f"wrote {out}")

# Also print the numbers so they can be quoted in the §4.9 prose
print("\nTop-N probabilities:")
print(f"{'word':<14} {'count':>6} {'uniform':>10} {'frequency':>12} "
      f"{'^{:.2f}'.format(EXPONENT):>10}")
for w, c, pu, pf, ps in zip(top_words, top_counts,
                            p_uniform, p_frequency, p_smoothed):
    print(f"{w:<14} {c:>6} {pu:>10.4f} {pf:>12.4f} {ps:>10.4f}")
