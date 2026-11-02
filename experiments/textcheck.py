import nunchi.eval_sentence as es
from nunchi.demo import compose
from nunchi.tagger import KO
def text_of(keys, r):
    lang, path = es.verdict(r)
    if lang == "split":
        path = list(path) + [path[-1]] * (len(keys) - len(path))
        out, start = "", 0
        for i in range(1, len(keys) + 1):
            if i == len(keys) or path[i] != path[start]:
                out += compose(keys[start:i]) if path[start] else keys[start:i]
                start = i
        return out
    return compose(keys) if lang == "ko" else keys


def expected(k, y):
    out, start = "", 0
    for i in range(1, len(k) + 1):
        if i == len(k) or y[i] != y[start]:
            out += compose(k[start:i]) if y[start] == KO else k[start:i]
            start = i
    return out


