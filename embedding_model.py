import math
import random

class EmbeddingModel:
    def __init__(self, vocab_size, embedding_dim=5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embeddings = []
        for _ in range(vocab_size):
            vector = []
            for _ in range(embedding_dim):
                vector.append(random.uniform(-0.5, 0.5))
            self.embeddings.append(vector)

    def get_vector(self, word_id):
        return self.embeddings[word_id]

    def train_on_pair(self, center_word, context_word, learning_rate=0.01):
        center_vector = self.embeddings[center_word]
        context_vector = self.embeddings[context_word]

        for i in range(self.embedding_dim):
            center_value = center_vector[i]
            context_value = context_vector[i]

            center_vector[i] = center_value + learning_rate * (context_value - center_value)
            context_vector[i] = context_value + learning_rate * (center_value - context_value)

    # Same arithmetic shape as train_on_pair but with a flipped sign so the two vectors move slightly further apart instead of closer together
    # Called on randomly sampled "negative" word ids to counteract the global
    def train_on_negative(self, center_word, negative_word, learning_rate=0.01):
        center_vector = self.embeddings[center_word]
        negative_vector = self.embeddings[negative_word]

        for i in range(self.embedding_dim):
            center_value = center_vector[i]
            negative_value = negative_vector[i]

            center_vector[i] = center_value - learning_rate * (negative_value - center_value)
            negative_vector[i] = negative_value - learning_rate * (center_value - negative_value)

    # Margin oriented push: only acts when the two vectors are within the defined margin
    # Euclidean distance of each other.
    # This replaces the above train_on_negative with an off-switch so once a pair is past margin, the update reduces to zero and stays there, so already-far-enough pairs no longer get pushed apart further

    def train_on_negative_margin(self, center_word, negative_word, margin, learning_rate=0.01):
        center_vector = self.embeddings[center_word]
        negative_vector = self.embeddings[negative_word]

        # Compute the difference vector and its Euclidean length once
        # Zip takes multiple iterables and walks through them side by side, so every value side by side and groups them by position
        diff = [c_value - n_value for c_value, n_value in zip(center_vector, negative_vector)]
        # Calculate the euclidean lenght of the diff vector
        distance = math.sqrt(sum(d * d for d in diff))

        # Off-switch: vectors already at least 'margin' apart -> no update
        if distance >= margin:
            return

        # Protects against 0 divison
        if distance < 1e-12:
            return

        # Push apart along (center - negative) / distance, scaled by how far below the margin the pair currently is.
        scale = (margin - distance) / distance
        for i in range(self.embedding_dim):

            # Push both the center vector and the sampled negative vector apart
            # moves "outward" along diff
            center_vector[i] += learning_rate * scale * diff[i]
            # moves "outward" along -diff
            negative_vector[i] -= learning_rate * scale * diff[i]