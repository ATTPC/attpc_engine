from attpc_engine.detector.pairing import pair, unpair


def test_pairing_low():
    x = 56
    y = 937
    id = y**2 + x
    id_u = pair(x, y)
    x_u, y_u = unpair(id_u)
    print(id, x_u, y_u)

    assert id_u == id
    assert x_u == x
    assert y_u == y


def test_pairing_hi():
    x = 937
    y = 56
    id = x**2 + x + y
    id_u = pair(x, y)
    x_u, y_u = unpair(id_u)

    assert id_u == id
    assert x_u == x
    assert y_u == y
