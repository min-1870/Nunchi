# Nunchi

[English](README.md) | **한국어**

> 한/영 키 안 눌러도 알아서 고쳐주는 Windows 입력 도우미

영어 모드에서 `dkssud`를 치면 `안녕`으로, 한글 모드에서 `ㅗ디ㅣㅐ`를 치면 `hello`로 바꿔줍니다. 입력 모드도 같이 바뀝니다.

> **다운로드:** [Windows 앱](https://github.com/min-1870/Nunchi/releases/latest) · [Hugging Face 모델](https://huggingface.co/min-1870/nunchi-tr5n-102k)
>
> **상태: 초기 릴리스 (0.1.0).** 앱에 코드 서명이 없어서, 실행할 때 Windows가 경고할 수 있습니다.
>
> - **모델:** TR5n-102K. 키 흐름 전체를 보는 Transformer 하나입니다(파라미터 320만 개, CPU 스레드 하나로 키 입력당 3.8 ms). 숫자가 든 단어(`3개`, `gpt4`)와 한 단어 안의 영어 + 조사(`MBC는`)도 처리합니다.
> - **문서 (영어):** [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) (현재 상태, 남은 문제, 모든 실험) · [docs/EVALUATION.md](docs/EVALUATION.md) (평가셋, 용어, 차이를 판정하는 방법) · [models/registry.json](models/registry.json) (모든 모델과 설정).

## 원칙

- **100% 로컬.** 네트워크를 전혀 쓰지 않습니다. 키 입력을 다루는 프로그램이라 신뢰가 먼저입니다.
- **확신 없으면 안 건드림.** 잘못 고치는 것이 못 고치는 것보다 비쌉니다.
- **되돌리기 단축키.** 되돌린 단어는 예외로 학습해서 다시는 고치지 않습니다.
- **단어 경계 후 100ms 안에 수정.**

## 동작 방식

```
키 입력 수집 (WH_KEYBOARD_LL)
  → 지금 줄의 키 흐름, 띄어쓰기 포함                          ... rmfjgek dlqkt rkqtl  MBCsms
  → TR5n: 키 흐름 전체를 보는 Transformer, 키마다 라벨          ... k k k k k k k  e e e k k k
  → 단어 중간: 아주 확실할 때만(0.99 이상) 미리 IME 전환
  → 스페이스: 확실하면(0.85 이상) 단어를 바꾸고, 단어 안에서 언어가 바뀌는
      경로가 확실하면(0.9 이상) 쪼갬:                          MBC + 는
  → 한 단어 뒤: 앞 단어를 다시 판단(0.8 이상), 앱이 바꾼 단어도 한 번은 되돌릴 수 있음
  → 사용자별: 짧고 애매한 단어(so / 내)는 그 사용자가 남긴 쪽으로 기움
  → 수정: Backspace → IME 전환 → raw 키 재입력
```

모든 판단은 화면 글자가 아니라 **raw 키 순서**(`dkssud`)로 합니다. 그래서 어느 모드에서 쳤든 똑같이 처리됩니다. 학습 라벨은 사람이 달 필요가 없습니다. 한글로 쓰인 글은 한국어, 알파벳으로 쓰인 글은 영어입니다. 기준값은 앱의 것으로, 배포 모델마다 한 번 앱 규칙대로 dev에서 정하고 test에서 한 번 확인합니다.

## 결과

학습에 쓰지 않은 평가셋의 dev 절반 기준이고, test 절반은 마지막 확인용으로 남겨 둡니다. 입력기처럼 모든 단어를 반드시 판단합니다. 스페이스에서 판단하고, 3단어 뒤에 95% 이상 확신하면서 2단어, 5단어 뒤에도 같은 답이면 고칩니다. 숫자는 칸별로 최종적으로 틀린 단어의 비율(%)입니다. 셋을 만드는 방법은 [docs/EVALUATION.md](docs/EVALUATION.md)에 있습니다. TR5n-102K는 무작위 seed만 바꿔 두 번 학습했고, 둘 다 적었습니다.

| 칸 | dev 단어 수 | TR3-51K | TR4-51K | **TR5n-102K** seed 0 | **TR5n-102K** seed 1 |
|---|---|---|---|---|---|
| 한국어 → 영어, 영어 구간의 시작 / 중간 / 끝 | 1,011 / 4,368 / 2,993 | 3.04 / 2.27 / 4.04 | 2.15 / 1.18 / 2.35 | 2.30 / 1.31 / 2.24 | 2.23 / 1.21 / 2.11 |
| 영어 → 한국어, 한국어 구간의 시작 / 중간 / 끝 | 1,595 / 2,090 / 1,465 | 1.26 / 5.42 / 12.16 | 0.22 / 1.78 / 4.64 | 0.29 / 2.45 / 5.20 | 0.29 / 1.77 / 4.54 |
| 한국어만 / 영어만 (본문 셋) | 60,267 / 39,011 | 0.12 / 0.04 | 0.11 / 0.04 | 0.10 / 0.03 | 0.09 / 0.05 |
| 한국어 단어이기도 한 영어 단어 (`so` = 내, `go` = 해) | 3,087 | 14.16 | 7.22 | 6.32 | 6.48 |
| 영어 단어이기도 한 한국어 단어 (해 = `go`) | 2,998 | | 1.95 | 2.06 | 1.99 |
| 한국어 글 속 자모 한 글자 (ㅋ, ㅠ) | 2,739 | | 7.73 | 7.87 | 7.33 |
| 한국어 표현과 키가 같은 영어 단어 (`gee` = ㅎㄷㄷ) | 547 | 19.41 | 11.16 | 12.09 | 12.09 |
| 띄어쓰기 없이 두 언어가 붙은 단어 (`QR코드`, `MBC는`) | 1,513 | | | 6.79 | 7.24 |
| 영어 글 속 숫자 단어 (`gpt4`, `5km`) | 2,993 | 3.38 | 0.73 | 0.33 | 0.40 |
| 오타 하나 / Shift를 계속 누른 채 (까를 `RK`) | 14,437 / 3,842 | | | 1.08 / 0.22 | 1.05 / 0.22 |
| 대문자 약어 (`EU`, `WHO`) | 1,504 | | | 0.20 | 0.30 |
| 단어 하나만: 한국어 / 영어 / 영어 + 조사 (`apk의`) | 2,000 / 1,993 / 965 | | | | 0.50 / 1.91 / 7.46 |

빈칸은 그 모델을 채점할 때 해당 셋이 없었다는 뜻입니다. 비교한 모델이 모두 채점된 133,050단어 기준으로는 TR4-51K 879개, TR5n-102K 840개(seed 0)와 806개(seed 1)가 틀렸습니다. 같은 레시피의 seed 두 개가 여기서 35~45개 차이 나므로, 그보다 작은 차이는 결과로 보지 않습니다.

**실제 사용에서:** 만든 사람이 32일 동안 Claude Code에 직접 친 17,876단어로 재면, 앱의 기준값에서 0.79%(약 126단어에 하나)가 틀립니다. 예전 기준값에서는 1.16%였습니다. 대부분 줄의 첫 단어(`so` → 내)와 조사가 붙은 소문자 영어(`wsl로`)입니다.

**속도:** Ryzen 5 5600 스레드 하나로 키 입력당 3.8 ms입니다(ONNX Runtime, 파라미터 320만 개). 키를 칠 때마다 지금 줄(최대 64키)을 한 번 읽습니다.

## 지금까지 알게 된 것

모델을 만든 여섯 가지 교훈입니다. 전체를 순서대로, 수치와 함께 보려면 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md#what-each-step-showed)(영어)를 보세요.

- **단어가 아니라 문장으로 잴 것.** 첫 모델은 단어 하나 기준 오수정이 0.02%였지만 실제 문장에서는 1.03%였습니다. 짧고 애매한 단어(그 `rm`, 내 `so`)는 단어 목록에는 드물고 실제 글에는 흔합니다.
- **데이터를 고친 것이 어떤 문맥 기법보다 효과가 컸습니다.** 잘못된 모드로 친 한국어를 거르려던 필터가 정확히 어려운 경우들(`so`, `go`, `B급`, 한 글자)을 지우고 있었습니다. 단어 빈도로 판단하게 바꾸자 돌아왔고, 다음 데이터 단계까지 더해 실제 섞인 문장 오류가 76에서 52로 줄었습니다.
- **키마다, 키 흐름 전체 위에서 라벨을 붙일 것.** 키 단위 태거는 단어 안에서 언어가 바뀌는 경우(`github` + `에`)를 다루고, 띄어쓰기까지 포함한 키 흐름 전체를 보는 Transformer 하나가 단어 요약 위의 모델보다 나았습니다.
- **학습이 타이핑과 같은 것을 보게 할 것.** 층을 쌓은 어텐션은 이웃을 거쳐 여러 단어 앞까지 닿지만, 타이핑에서는 많아야 한 단어 앞만 보입니다. 그래서 판단하는 단어 하나에만 정답을 붙여 학습합니다.
- **seed가 대부분의 설정보다 결과를 더 흔듭니다.** 같은 레시피도 seed만 바꾸면 51k 스텝에서 44단어가 차이 납니다. 그래서 모든 변경은 두 번 학습해서 칸별로 판단합니다.
- **남은 오류는 양쪽 방향으로 틀립니다.** 같은 문자열이 셋에 따라 반대로 틀리고(ㅎㅎ를 `gg`로, `gee`를 ㅎㄷㄷ로; 해를 `go`로, `go`를 해로), 이걸 노린 데이터 변경은 모두 한쪽 오류를 반대쪽으로 옮기기만 했습니다.

## 재현

Windows에서 Python 3.12로 테스트했고, 명령은 Git Bash 기준입니다. 콘솔에 한글이 출력되도록 `PYTHONUTF8=1`이 필요합니다.

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
mkdir -p data/raw/kcbert data/raw/mc4 data/raw/wiktionary data/raw/imdb data/raw/goemotions data/raw/wikitext103 data/raw/reddit_big
curl -L -o data/raw/nsmc_train.txt https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt
curl -L -o data/raw/nsmc_test.txt https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt
curl -L -o data/raw/eng_sentences.tsv.bz2 https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2
curl -L -o data/raw/kor_sentences.tsv.bz2 https://downloads.tatoeba.org/exports/per_language/kor/kor_sentences.tsv.bz2
for p in aa ab ac; do curl -L -o data/raw/kcbert/kcbert-train.tar.gz$p https://github.com/Beomi/KcBERT/releases/download/TrainData_v1/kcbert-train.tar.gz$p; done
curl -L -o data/raw/mc4/c4-ko-validation.json.gz https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/c4-ko-validation.tfrecord-00000-of-00001.json.gz
curl -L -o data/raw/wiktionary/kaikki-korean.jsonl https://kaikki.org/dictionary/Korean/kaikki.org-dictionary-Korean.jsonl
for s in train test; do curl -L -o data/raw/imdb/$s.parquet https://huggingface.co/api/datasets/stanfordnlp/imdb/parquet/plain_text/$s/0.parquet; done
for s in train validation test; do curl -L -o data/raw/goemotions/$s.parquet https://huggingface.co/api/datasets/google-research-datasets/go_emotions/parquet/simplified/$s/0.parquet; done
for i in 0 1; do curl -L -o data/raw/wikitext103/train_$i.parquet https://huggingface.co/api/datasets/Salesforce/wikitext/parquet/wikitext-103-raw-v1/train/$i.parquet; done
for s in buildapc gaming relationship_advice todayilearned; do curl -L -o data/raw/reddit_big/${s}_0.parquet https://huggingface.co/api/datasets/HuggingFaceGECLM/REDDIT_comments/parquet/default/$s/0.parquet; done
export PYTHONUTF8=1
.venv/Scripts/python -m nunchi.count_corpus     # 출처별 단어 / 영어 / 영어+한글 개수
.venv/Scripts/python -m nunchi.make_data        # 단어 train/valid/test
.venv/Scripts/python -m nunchi.make_noise       # 오타/코드 test셋, 오타 학습 데이터
.venv/Scripts/python -m nunchi.sentence_data    # 출처별 문장 (data/sent)
.venv/Scripts/python -m nunchi.web_mixed train  # mC4 train 파일 0~9번 (자동 다운로드) -> 섞인 문장
.venv/Scripts/python -m nunchi.web_mixed        # mC4 validation -> 웹 평가셋
.venv/Scripts/python -m nunchi.web_mixed particle
.venv/Scripts/python -m nunchi.translations     # 번역 치환 문장, 양방향
.venv/Scripts/python -m nunchi.ambiguous_eval   # 키가 양쪽 언어에서 흔한 단어의 평가셋
```

v3 학습 풀과 평가셋은 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)의 "How to run" 명령으로 만듭니다. 모델은 PyTorch로 학습합니다. CPU라면 같은 venv에 `pip install torch`로 충분하고, Windows의 AMD GPU라면 대신 AMD의 ROCm 패키지를 설치합니다([AMD 안내](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)). TR5n-102K 학습(Radeon RX 9060 XT에서 약 2시간 반)과 채점:

```bash
.venv/Scripts/python -u experiments/run.py experiments/configs/tr5n_102k.json
.venv/Scripts/python -u experiments/system_eval.py TR5n-102K=TR5n-102K
```

중간에 멈춘 학습은 `NUNCHI_RESUME=1`로 체크포인트에서 이어갑니다.

**사용한 버전:** [requirements-train.txt](requirements-train.txt)(학습, ROCm), [requirements-export.txt](requirements-export.txt)(ONNX 변환), 원본 파일별 해시는 [data_versions.sha256](data_versions.sha256)에 있습니다(2026-10-31 ~ 2026-11-03에 받은 파일). Tatoeba와 kaikki.org는 시간이 지나면 내용이 바뀌므로 나중에 받으면 다를 수 있습니다. 데이터 워커와 GPU 연산이 결정적이지 않아서, 다시 학습한 모델도 조금씩 다릅니다.

## 구성

| 파일 | 역할 |
|---|---|
| `nunchi/dubeolsik.py` | 한글 → 두벌식 키 변환, 한글 모드로 읽었을 때의 음절 모양 |
| `nunchi/count_corpus.py` | 출처별 개수: 한국어 단어, 한국어 글 속 영어, 영어 + 조사 |
| `nunchi/make_data.py`, `make_noise.py` | 단어 데이터, 오타 셋 |
| `nunchi/sentence_data.py` | 출처별 문장, 잘못된 모드로 친 한국어를 빈도로 거르는 필터 |
| `nunchi/web_mixed.py` | mC4 한국어: train 분할로 학습 풀, validation 분할로 평가셋 |
| `nunchi/translations.py` | 위키낱말사전 뜻풀이로 만든 번역 치환 문장 (외래어 포함) |
| `nunchi/loanwords.py` | 위키낱말사전 어원에서 뽑은 외래어 ↔ 영어 쌍, 동음이의어 제외 |
| `nunchi/ambiguous_eval.py` | 키가 양쪽 언어에서 흔한 단어의 평가셋 |
| `nunchi/tagger.py` | 키 단위 BiLSTM-CRF 단어 태거 (HMM 기준선), CRF, 영어 + 조사 단어 |
| `nunchi/sentence_tagger.py` | 학습 데이터 샘플러 (역할별 비율), LSTM 문장 모델 |
| `nunchi/stream_tagger.py` | 키 흐름 Transformer (TX, RV, TR), 이어하기 되는 학습, `decide()` |
| `nunchi/models.py` | `models/registry.json`에서 이름으로 모델과 문맥, 기준값 불러오기 |
| `nunchi/eval_sentence.py` | 문장의 모든 단어 판단 (바로 / 한 단어 뒤), 단어 쪼개기 |
| `nunchi/export_onnx.py` | 앱용 ONNX 변환 |
| `nunchi/context.py` | HMM 문맥 실험, 실제 섞인 리뷰를 포함한 문장 셋 |
| `nunchi/demo.py`, `typebench.py` | 로컬 테스트 페이지, 실제 키 입력 벤치마크 |
| `experiments/` | 숫자를 살린 평가셋, 종류별 표, 판단 기준 곡선, 설정 파일로 학습을 돌리는 `run.py` |
| `experiments/system_eval.py` | 후보정 규칙까지 넣어 모든 dev / test 셋으로 모델 채점 |
| `experiments/logs/app_tune.py`, `early_tune.py` | 앱 기준값을 앱 규칙 그대로 채점 |
| `models/registry.json` | 모든 모델: 부모, 설정, 상태 |
| `docs/` | 실험 기록과 평가 가이드 |
| `app/` | Windows 앱 |

## 로드맵

- [x] 한글 → 두벌식 키, fastText 기준선, HMM 문맥
- [x] 실제 섞인 문장 평가셋 (NSMC 리뷰, mC4 웹 글), 학습과 분리, 중복 제거
- [x] 키 단위 라벨링 모델 (단어 안 전환: `MBC는`, `github에`)
- [x] 키 흐름 위의 문장 모델, 타이핑과 같은 조건의 학습
- [x] 데이터: 빈도 기반 잘못된 모드 필터, mC4, 번역 치환, 소문자 증강, Shift 누른 채 오타
- [x] ONNX 변환
- [x] 바로 판단과 후보정을 같이 하는 한 모델 (TR)
- [x] 같은 설정 재학습의 흔들림 측정 (작은 차이를 판단하기 위해)
- [x] 더 길게 학습 (102k 스텝, 204k는 더 나아지지 않음)
- [x] 전환 지점별, 오타, 양방향 애매한 단어, 한 단어 입력 평가셋
- [x] 실제 타이핑으로 측정 (한 사람의 Claude Code 글)
- [ ] 깜빡임 없는 단어 중간 전환 (`wha` → 좀)
- [ ] 여러 사람의 타이핑으로 실제 사용 기준의 오류 비중 재기
- [x] Windows 앱: 키보드 훅, IME 제어, 예외 처리, 되돌리기, 후보정, 습관 반영 (0.1.0)

## 데이터 출처

- [NSMC](https://github.com/e9t/nsmc) (네이버 영화 리뷰): 한국어, CC0
- [KcBERT 사전학습 말뭉치](https://github.com/Beomi/KcBERT) (뉴스 댓글): 한국어, CC BY-SA 4.0
- [mC4](https://huggingface.co/datasets/allenai/c4) 한국어 (Common Crawl 웹 글): ODC-BY, 그리고 Common Crawl 이용 약관
- [Tatoeba](https://tatoeba.org): 영어, 한국어 문장, CC BY 2.0 FR
- [위키낱말사전](https://en.wiktionary.org) 한국어 항목, [kaikki.org](https://kaikki.org) 경유 (Tatu Ylonen, "Wiktextract: Wiktionary as Machine-Readable Structured Data", LREC 2022): CC BY-SA 4.0 / GFDL
- [wordfreq](https://github.com/rspeer/wordfreq): 영어 단어와 빈도, 데이터는 CC BY-SA 4.0
- [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) (위키백과 문서): 영어, CC BY-SA. 학습에 사용.
- [REDDIT_comments](https://huggingface.co/datasets/HuggingFaceGECLM/REDDIT_comments) (서브레딧 4개): 영어, 라이선스 표기 없음. 학습 샘플의 약 0.7%와 일부 평가셋에 사용.
- [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) (Reddit 댓글): 데이터 카드 기준 Apache-2.0. 평가셋과 소수의 학습 문장에 사용.
- [IMDB 리뷰](https://huggingface.co/datasets/stanfordnlp/imdb): 라이선스 "other", 데이터 카드에 조건 없음. 평가셋과 소수의 학습 문장에 사용.

이 저장소에는 위 출처에서 나온 데이터가 없습니다. 학습된 가중치도 저장소에 없습니다.

## 라이선스

코드는 [MIT 라이선스](LICENSE)입니다.

학습된 가중치는 앱과 함께 따로 공개하며, 앱 체험과 연구를 위해 있는 그대로 제공합니다. 가중치는 위 출처들로 학습했고, 각 출처의 조건이 적용됩니다. 데이터 일부가 CC BY-SA이므로, 가중치는 저작자가 허락할 수 있는 범위에서 CC BY-SA 4.0으로 제공합니다. 제3자의 글에 대한 권리는 허락하지 않습니다.

모든 출처에 라이선스가 표기돼 있지는 않습니다. 이 데이터에 권리가 있고 사용에 반대하신다면 이슈를 남겨 주세요. 해당 데이터를 빼고 다시 학습하거나 모델을 내리겠습니다.
