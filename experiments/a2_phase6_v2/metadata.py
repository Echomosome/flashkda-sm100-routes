"""序列地址表：CPU 元数据，供融合实现和离线检查共用。"""
def metadata(lengths, time, batch, chunk):
    """只有序列长度/地址的 CPU 元数据；不计算任何输入数值，放在 Graph 外创建。"""
    ls = list(lengths) if lengths is not None else [time] * batch
    if not ls or min(ls) <= 0 or (lengths is not None and (batch != 1 or sum(ls) != time)):
        raise ValueError('invalid positive sequence lengths')
    starts, counts, prefix, bos = [], [], [0], 0
    for n in ls:
        for off in range(0, n, chunk):
            starts.append(bos + off); counts.append(min(chunk, n - off))
        prefix.append(len(starts)); bos += n
    return starts, counts, prefix

