"""One structured JSON record per line, so logs stay machine-checkable."""

import json


def emit(tag, **fields):
    print(tag + " " + json.dumps(fields, ensure_ascii=False, sort_keys=True), flush=True)
