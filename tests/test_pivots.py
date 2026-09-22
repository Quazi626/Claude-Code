from flux.data.pivots import compute_floor_pivots


def test_classic_floor_pivot_formula():
    # H=110, L=90, C=100 -> P=100
    pivots = compute_floor_pivots(high=110, low=90, close=100)
    assert pivots.p == 100
    assert pivots.r1 == 110       # 2P - L = 200 - 90
    assert pivots.s1 == 90        # 2P - H = 200 - 110
    assert pivots.r2 == 120       # P + (H-L) = 100 + 20
    assert pivots.s2 == 80        # P - (H-L) = 100 - 20
    assert pivots.r3 == 130       # H + 2(P-L) = 110 + 2*10
    assert pivots.s3 == 70        # L - 2(H-P) = 90 - 2*10


def test_levels_sorted_ascending():
    pivots = compute_floor_pivots(high=110, low=90, close=100)
    values = [v for _, v in pivots.levels_sorted()]
    assert values == sorted(values)
