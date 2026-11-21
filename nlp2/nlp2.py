from pathlib import Path
import gensim
import numpy as np
MODEL_PATH = Path(__file__).with_name("cbow.txt")

TARGETS = ["селезень_NOUN", "система_NOUN"]
TOP_N = 10
NEIGHBORS_COUNT = 5
C_COUNT = 7
def clean(word):
    return word.removesuffix("_NOUN")

def top_indices(scores, count):
    indices = np.argpartition(scores, -count)[-count:]
    return indices[np.argsort(scores[indices])[::-1]]

def find_combination(model):
    words = [
        word for word in model.index_to_key
        if word.endswith("_NOUN")
    ]
    indices = [model.key_to_index[word] for word in words]
    vectors = model.vectors[indices].astype(np.float32)
    vectors /= np.maximum(
        np.linalg.norm(vectors, axis=1, keepdims=True),
        1e-12
    )
    word_to_id = {word: i for i, word in enumerate(words)}

    t1 = word_to_id[TARGETS[0]]
    t2 = word_to_id[TARGETS[1]]

    targets = [t1, t2]

    def nearest_to_target(target):
        scores = vectors @ vectors[target]
        scores[targets] = -np.inf

        return top_indices(scores, NEIGHBORS_COUNT)

    a_words = nearest_to_target(t1)
    b_words = nearest_to_target(t2)

    print("\nблизкие слова для первой цели:")
    print(", ".join(clean(words[i]) for i in a_words))
    print("\nблизкие слова для второй цели:")
    print(", ".join(clean(words[i]) for i in b_words))
    pairs = [
        (int(a), int(b))
        for a in a_words
        for b in b_words
    ]

    sums = np.array(
        [vectors[a] + vectors[b] for a, b in pairs],
        dtype=np.float32
    )
    target_vector = vectors[t1] + vectors[t2]

    coefficients = [0.60, 0.80, 1.00, 1.15]

    directions = np.concatenate([
        sums - k * target_vector
        for k in coefficients
    ])

    combinations = []

    for start in range(0, len(directions), 128):
        batch = directions[start:start + 128]

        similarities = vectors @ batch.T

        for j in range(similarities.shape[1]):
            pair_id = (start + j) % len(pairs)
            a, b = pairs[pair_id]

            scores = similarities[:, j]
            scores[[a, b, t1, t2]] = -np.inf

            c_words = top_indices(scores, C_COUNT)

            for c in c_words:
                combinations.append((a, b, int(c)))
    combinations = list(dict.fromkeys(combinations))
    print(f"\nПолучено комбинаций: {len(combinations)}")

    best = None
    best_quality = None
    successful = 0

    for start in range(0, len(combinations), 128):
        batch = combinations[start:start + 128]

        result_vectors = np.array([
            vectors[a] + vectors[b] - vectors[c]
            for a, b, c in batch
        ], dtype=np.float32)
        result_vectors /= np.maximum(
            np.linalg.norm(
                result_vectors, axis=1, keepdims=True
            ),
            1e-12
        )
        similarities = vectors @ result_vectors.T
        for j, (a, b, c) in enumerate(batch):
            scores = similarities[:, j]
            scores[[a, b, c]] = -np.inf

            top10 = top_indices(scores, TOP_N)

            if t1 not in top10 or t2 not in top10:
                continue

            successful += 1
            rank1 = int(np.where(top10 == t1)[0][0]) + 1
            rank2 = int(np.where(top10 == t2)[0][0]) + 1

            relations = vectors[top10] @ vectors[targets].T
            groups = np.argmax(relations, axis=1)

            counts = [
                int(np.sum(
                    (groups == k) &
                    (relations[:, k] > 0.30)
                ))
                for k in range(2)
            ]
            quality = (
                min(counts),
                sum(counts),
                -max(rank1, rank2),
                -(rank1 + rank2),
                min(float(scores[t1]), float(scores[t2]))
            )
            if best_quality is None or quality > best_quality:
                best_quality = quality
                best = (a, b, c)
    print(f"Успешных комбинаций: {successful}")
    return (
        [words[i] for i in best]
        if best is not None else None
    )
def nearest_nouns(model, positive, negative):
    limit = 100
    while True:
        neighbors = model.most_similar(
            positive=positive,
            negative=negative,
            topn=limit
        )
        nouns = [
            (word, score)
            for word, score in neighbors
            if word.endswith("_NOUN")
        ]
        if len(nouns) >= TOP_N:
            return nouns[:TOP_N]

        if limit >= len(model):
            raise ValueError("Недостаточно существительных")
        limit = min(limit * 2, len(model))

def main():
    model = gensim.models.KeyedVectors.load_word2vec_format(
        str(MODEL_PATH),
        binary=False
    )
    for target in TARGETS:
        if target not in model:
            raise ValueError(f"Нет слова в модели: {target}")

    combination = find_combination(model)

    if combination is None:
        print("Подходящая комбинация не найдена.")
        return

    a, b, c = combination

    positive = [a, b]
    negative = [c]
    results = nearest_nouns(model, positive, negative)
    print("найденная кобинация:")
    print(f"{clean(a)} + {clean(b)} - {clean(c)}")
    print("\nтоп-10 ближайших соседей:")
    for rank, (word, score) in enumerate(results, 1):
        mark = "  <--ЦЕЛЬ" if word in TARGETS else ""
        print(
            f"{rank:2}. {clean(word):22} "
            f"{score:.4f}{mark}"
        )
    
    ranks = {
        word: i
        for i, (word, _) in enumerate(results, 1)
    }
    for target in TARGETS:
        print(
            f"{clean(target)}: "
            f"место {ranks.get(target, 'вне десятки')}"
        )
if __name__ == "__main__":
    main()
