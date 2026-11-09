"""题目 baseline / FLA 的固定 SHA；来源沿用已完成对拍的 A2。"""

SOURCE_SHA256 = {
    "fwd.h": "462afefb5c1bfe89428166cd81fc70954606049a41a4b67376025a2b2f91d7cc",
    "flash_kda.cpp": "f736ab87c7c2e47fe416f8d04d62502a257a6ff32a0b23452c07e1bc65c87c04",
    "smxx/fwd_kernel1.cuh": "e0f369b51753d5e1a62b8435f0b692fdc856d87e52eb27d012c22b6575a0370f",
    "smxx/fwd_kernel2.cuh": "f8a634154ca26a18855adbd3db17d6208a874b16f173d0eb9814bfefff03b9b9",
    "smxx/fwd_launch.cu": "dbaf9ba867fc8c799cf147b7e17bd2aba4a08b1b63c5f7df0cca1173a0df5562",
    "smxx/utils.cuh": "211cc28d910758e2658a53f5bcda779f1076df281ff75c09ce9bd1638a54151a",
}

FLA_NAIVE_SHA256 = "60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016"

def replace_exact(text, old, new, count=1):
    """每处变更检查命中数；源码不同即停下，避免漏改或误改 key 维。"""
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(f"patch anchor count {actual} != {count}: {old[:140]!r}")
    return text.replace(old, new)
