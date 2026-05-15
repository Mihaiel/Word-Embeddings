# Frequency-biased negative sampler used by §4.9.
#
# Builds a probability distribution over the vocabulary where each word's
# probability is proportional to its corpus count raised to a power (the
# "exponent"). With exponent = 0 you recover the uniform distribution from
# §4.7 / §4.8. With exponent = 1 you get pure frequency-proportional
# sampling. The default 0.75 is the empirical compromise reported by
# Mikolov et al. 2013 — a tuning knob, not a derived principle.
#
# Sampling itself uses a cumulative-distribution array plus binary search,
# which gives O(log V) sampling without any external dependencies. For
# vocabularies in the thousands this is fast enough that the cost is
# invisible inside the training loop.

import bisect
import random


class UnigramSampler:
    """Sample word ids in proportion to count^exponent.

    Parameters
    ----------
    word_counts : dict[int, int]
        Mapping word_id -> number of times that word appears in the corpus.
        Produced by `TextPreprocessor.count_words` (or any equivalent).
    exponent : float, default 0.75
        Smoothing exponent. 0 = uniform, 1 = pure frequency, 0.75 is the
        word2vec-style compromise that prevents very common words from
        dominating the distribution.
    """

    def __init__(self, word_counts, exponent=0.75):
        # Stable ordering of word ids so the cumulative array is reproducible
        self.word_ids = sorted(word_counts.keys())

        # Probabilities are proportional to count^exponent. We do not
        # normalise to sum to 1; the cumulative array carries the scale and
        # `random.uniform(0, total)` lines up with it directly.
        self.weights = [word_counts[wid] ** exponent for wid in self.word_ids]

        # Cumulative distribution: cumulative[i] = sum of weights[0..i].
        # `bisect_right(cumulative, x)` returns the index of the first
        # entry strictly greater than x, which is exactly the sampled
        # word's index in self.word_ids.
        self.cumulative = []
        running = 0.0
        for w in self.weights:
            running += w
            self.cumulative.append(running)
        self.total = running

        self.exponent = exponent

    def sample(self, exclude_id=None):
        """Draw a single word id from the distribution.

        If `exclude_id` is given (typically the current center word), the
        sampler rerolls until it draws something different. This loop
        terminates almost immediately on any reasonable vocabulary because
        the excluded id has probability proportional to its count, which
        is at most a small fraction of the total.
        """
        while True:
            x = random.uniform(0.0, self.total)
            idx = bisect.bisect_right(self.cumulative, x)
            # Guard the upper boundary: bisect_right can in principle
            # return len(cumulative) when x == total, which is rare but
            # not impossible.
            if idx >= len(self.word_ids):
                idx = len(self.word_ids) - 1
            sampled = self.word_ids[idx]
            if sampled != exclude_id:
                return sampled

    def probability(self, word_id):
        """Return the sampling probability of a single word id.

        Useful for the §4.9 figure that compares uniform / pure-frequency /
        smoothed distributions side by side.
        """
        return self.weights[self.word_ids.index(word_id)] / self.total
