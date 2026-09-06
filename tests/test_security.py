from pulsar.security import hash_api_key, new_api_key


def test_key_is_random_and_hashed():
    raw1, record1 = new_api_key("one")
    raw2, record2 = new_api_key("two")
    assert raw1.startswith("pulsar_live_")
    assert raw1 != raw2
    assert record1["key_hash"] == hash_api_key(raw1)
    assert raw1 not in str(record1)
    assert record1["key_hash"] != record2["key_hash"]
