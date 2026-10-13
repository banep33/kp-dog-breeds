"""Testovi za deduplikaciju. Union-Find je logika od koje zavisi poštenje SVIH metrika
(ako grupisanje ne radi, duplikati cure iz train u test) -> zaslužuje pravi test."""
import numpy as np
import pandas as pd

from dogbreeds.dedup import UnionFind, build_groups, cross_label_pairs, find_similar_pairs


def test_union_find_transitivity():
    # A~B, B~C -> A,B,C u istoj grupi; D, G sami; E~F zajedno
    g = build_groups([(0, 1), (1, 2), (4, 5)], 7)
    assert g[0] == g[1] == g[2]
    assert g[4] == g[5]
    assert len({g[0], g[3], g[4], g[6]}) == 4  # 4 različite grupe
    assert len(g) == 7


def test_union_find_long_chain():
    """Path compression mora da izdrži dugačak lanac bez rekurzije."""
    n = 50_000
    g = build_groups([(k, k + 1) for k in range(n - 1)], n)
    assert len(set(g.tolist())) == 1  # sve u jednoj grupi


def test_union_find_ids_are_compact():
    g = build_groups([(0, 1)], 5)
    assert sorted(set(g.tolist())) == [0, 1, 2, 3]  # 0..G-1, bez praznina


def test_union_by_size_and_idempotent_union():
    uf = UnionFind(4)
    assert uf.union(0, 1) is True
    assert uf.union(0, 1) is False  # već spojeni
    uf.union(2, 3)
    uf.union(1, 2)
    assert len({uf.find(k) for k in range(4)}) == 1
    assert uf.size[uf.find(0)] == 4


def test_find_similar_pairs_detects_known_duplicates():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(5, 16)).astype(np.float32)
    emb = np.vstack([base, base[0] + 0.001 * rng.normal(size=16)])  # 6. je skoro kopija 1.
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    pairs, nn_sim = find_similar_pairs(emb, min_sim=0.9, block=2)
    assert {(0, 5)} == {(int(r.i), int(r.j)) for r in pairs.itertuples()}
    assert pairs["sim"].iloc[0] > 0.99
    assert nn_sim[0] > 0.99 and nn_sim[5] > 0.99
    assert len(nn_sim) == 6
    assert (pairs["i"] < pairs["j"]).all()  # samo gornji trougao, nema duplih parova


def test_find_similar_pairs_block_size_does_not_change_result():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(40, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    a, _ = find_similar_pairs(emb, min_sim=0.5, block=7)
    b, _ = find_similar_pairs(emb, min_sim=0.5, block=40)
    assert set(zip(a.i, a.j)) == set(zip(b.i, b.j))


def test_cross_label_pairs_flags_label_mismatch():
    pairs = pd.DataFrame({"i": [0, 1], "j": [2, 3], "sim": [0.99, 0.94]})
    breeds = np.array(["beagle", "pug", "beagle", "boxer"])
    out = cross_label_pairs(pairs, breeds, min_sim=0.93)
    assert len(out) == 1 and out.loc[0, "breed_i"] == "pug" and out.loc[0, "breed_j"] == "boxer"
