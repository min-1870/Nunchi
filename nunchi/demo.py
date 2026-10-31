"""Local test page: type a sentence (either IME mode) and see what the model would do per word.

    .venv/Scripts/python -m nunchi.demo      # then open http://localhost:8765

Each word is turned into raw keys (Hangul -> 두벌식 keys, Latin stays), scored by fastText,
then passed through the chosen context model. The Korean reading uses a simple greedy
두벌식 composer for display only (not the IME-exact FSM on the roadmap).
"""
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import fasttext

from nunchi.context import Scorer, decode, four_state, two_state
from nunchi.dubeolsik import CHO, COMPAT, JONG, JUNG, to_keys
from nunchi.train_fasttext import MODELS

PORT = 8765
T = 2
MODES = {
    "none": (two_state(0.5), "filter"),
    "2state": (two_state(0.7), "hybrid"),
    "4state": (four_state(0.99, 0.9, 0.7), "hybrid"),
}
CORE = re.compile(r"^([^\w가-힣ㄱ-ㅣ]*)(.*?)([^\w가-힣ㄱ-ㅣ]*)$")

_LONE = {k: ch for ch, k in COMPAT.items()}
_CONS = set(CHO)
_VOWEL1 = {j for j in JUNG if len(j) == 1}


def compose(keys: str) -> str:
    """Greedy 두벌식 composition of a key sequence, for display."""
    keys = "".join(k if k in "QWERTOP" else k.lower() for k in keys)
    out, cho, jung, jong = [], "", "", ""

    def flush():
        nonlocal cho, jung, jong
        if cho and jung:
            out.append(chr(0xAC00 + (CHO.index(cho) * 21 + JUNG.index(jung)) * 28 + JONG.index(jong)))
        else:
            out.extend(_LONE[x] for x in (cho, jung) if x)
        cho = jung = jong = ""

    for k in keys:
        if k in _CONS:
            if cho and jung and not jong and k in JONG:
                jong = k
            elif cho and jung and jong and jong + k in JONG:
                jong += k
            else:
                flush()
                cho = k
        elif k in _VOWEL1:
            if cho and jung and jong:
                move = jong[-1] if len(jong) == 2 else jong
                jong = jong[:-1] if len(jong) == 2 else ""
                flush()
                cho, jung = move, k
            elif jung and jung + k in JUNG:
                jung += k
            elif cho and not jung:
                jung = k
            else:
                flush()
                jung = k
        else:
            flush()
            out.append(k)
    flush()
    return "".join(out)


class Model:
    def __init__(self):
        self.scorer = Scorer(fasttext.load_model(str(MODELS / "ft_shape_aug.ftz")))

    def run(self, text, mode, threshold):
        words = []
        for tok in text.split():
            pre, core, post = CORE.match(tok).groups()
            if core:
                words.append({"tok": tok, "pre": pre, "core": core, "post": post, "keys": to_keys(core)})
        if not words:
            return {"words": [], "output": ""}
        zs = [self.scorer.z(w["keys"]) for w in words]
        model, how = MODES[mode]
        alone, _ = decode(zs, two_state(0.5), T, "filter")
        probs, retro = decode(zs, model, T, how)
        out = []
        for w, a, p, r in zip(words, alone, probs, retro):
            conf = max(p, 1 - p)
            decided = ("ko" if p >= 0.5 else "en") if conf >= threshold else None
            ko_read, en_read = compose(w["keys"]), w["keys"]
            shown = {"ko": ko_read, "en": en_read}.get(decided, w["core"])
            w.update(p_word=a, p=p, retro=r and decided is not None, decided=decided,
                     ko=ko_read, en=en_read, out=w["pre"] + shown + w["post"],
                     changed=decided is not None and shown != w["core"])
            out.append(w["out"])
        return {"words": words, "output": " ".join(out)}


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Nunchi Demo</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--line:#e7e5e4;--card:#fff;--ko:#2563eb;--en:#16a34a;--un:#a8a29e;--chg:#fef3c7}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--line:#44403c;--card:#292524;--ko:#60a5fa;--en:#4ade80;--un:#78716c;--chg:#713f12}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Malgun Gothic",sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px}h1{font-size:20px;margin:0 0 4px}p.sub{color:var(--mut);margin:0 0 16px}
textarea{width:100%;min-height:90px;padding:12px;font:16px/1.5 inherit;font-family:inherit;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
.row{display:flex;gap:16px;flex-wrap:wrap;align-items:center;margin:12px 0;color:var(--mut);font-size:14px}
select{padding:4px 8px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
.out{padding:12px;border-radius:8px;background:var(--card);border:1px solid var(--line);font-size:18px;min-height:48px;margin-bottom:16px}
.out span.chg{background:var(--chg);border-radius:4px;padding:0 2px}
.tbl{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--mut);font-weight:500}
code{font-family:ui-monospace,Consolas,monospace}.ko{color:var(--ko);font-weight:600}.en{color:var(--en);font-weight:600}.un{color:var(--un)}
.bar{display:inline-block;width:60px;height:8px;border-radius:4px;background:linear-gradient(90deg,var(--ko) var(--p),var(--en) var(--p));vertical-align:middle;margin-right:6px}
</style></head><body><main>
<h1>Nunchi 테스트</h1>
<p class="sub">한글 모드든 영어 모드든 그냥 쳐보세요. 단어마다 키로 바꿔서 판정하고, 앱이 바꿀 결과를 보여줘요.</p>
<textarea id="t" placeholder="예: dkssudgktpdy 오늘 ㅇ데ㅣㅐㅛ 했는데 TV qhkTdj" autofocus></textarea>
<div class="row">
<label>문맥 <select id="m"><option value="2state">2상태 HMM (현재)</option><option value="none">없음</option><option value="4state">4상태</option></select></label>
<label>고치는 기준 <select id="th"><option>0.99</option><option>0.9</option><option>0.999</option></select></label>
</div>
<div class="out" id="o"></div>
<div class="tbl"><table><thead><tr><th>입력</th><th>키</th><th>한글로 읽으면</th><th>영어로 읽으면</th><th>단어만 P(한)</th><th>문맥 반영 P(한)</th><th>판정</th></tr></thead><tbody id="b"></tbody></table></div>
</main><script>
const $=id=>document.getElementById(id);let timer;
const pct=p=>(p*100).toFixed(p>0.999||p<0.001?2:1)+'%';
const esc=s=>s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function run(){
  const r=await fetch('/api',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({text:$('t').value,mode:$('m').value,threshold:+$('th').value})});
  const d=await r.json();
  $('o').innerHTML=d.words.map(w=>w.changed?`<span class="chg">${esc(w.out)}</span>`:esc(w.out)).join(' ');
  $('b').innerHTML=d.words.map(w=>{
    const lab=w.decided?`<span class="${w.decided}">${w.decided==='ko'?'한국어':'영어'}</span>${w.retro?' <span class="un">(한 단어 뒤)</span>':''}`:'<span class="un">안 고침</span>';
    return `<tr><td>${esc(w.tok)}</td><td><code>${esc(w.keys)}</code></td><td>${esc(w.ko)}</td><td>${esc(w.en)}</td>
    <td><span class="bar" style="--p:${w.p_word*100}%"></span>${pct(w.p_word)}</td>
    <td><span class="bar" style="--p:${w.p*100}%"></span>${pct(w.p)}</td><td>${lab}</td></tr>`}).join('');
}
const later=()=>{clearTimeout(timer);timer=setTimeout(run,150)};
$('t').addEventListener('input',later);$('m').addEventListener('change',run);$('th').addEventListener('change',run);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    model = None

    def do_GET(self):
        self._send(200, "text/html; charset=utf-8", PAGE.encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        mode = body.get("mode") if body.get("mode") in MODES else "2state"
        res = self.model.run(body.get("text", ""), mode, float(body.get("threshold", 0.99)))
        self._send(200, "application/json", json.dumps(res, ensure_ascii=False).encode())

    def _send(self, code, ctype, data):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    Handler.model = Model()
    print(f"Nunchi demo on http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
