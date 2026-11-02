# Nunchi (gksdud)

[English](README.md) | **한국어**

> 한/영 키 안 눌러도 알아서 고쳐주는 Windows 입력 도우미

영어 모드에서 `dkssud`를 치면 `안녕`으로, 한글 모드에서 `ㅗ디ㅣㅐ`를 치면 `hello`로 바꿔줍니다. 입력 모드도 같이 바뀝니다.
저장소 이름 `gksdud`는 영어 모드에서 "한영"을 친 모양입니다.

> **상태: 연구 단계.** 기준 모델(TR3-long)은 스트림 Transformer 하나로 Space에서 단어를 판단하고, 단어가 더 쳐지면 다시 고칩니다. 숫자가 든 단어(`3개`, `gpt4`)와 한 단어 안의 영어 + 조사(`MBC는`)를 처리합니다. 이전 구성인 TX4(판단) + RV1(후보정)은 아래에 정리돼 있습니다. Windows 앱은 `app/`에서 만드는 중입니다. 진행 중인 실험과 각 실험에서 알게 된 것, 다음 할 일은 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), 평가셋과 차이를 판정하는 방법은 [docs/EVALUATION.md](docs/EVALUATION.md)에 있습니다.

## 원칙

- **100% 로컬.** 네트워크를 전혀 쓰지 않습니다. 키 입력을 다루는 프로그램이라 신뢰가 먼저입니다.
- **확신 없으면 안 건드림.** 잘못 고치는 것이 못 고치는 것보다 비쌉니다.
- **되돌리기 단축키.** 되돌린 단어는 예외로 학습해서 다시는 고치지 않습니다.
- **단어 경계 후 100ms 안에 수정.**

## 동작 방식

```
키 입력 수집 (WH_KEYBOARD_LL)
  → 최근 단어들의 키 흐름, 띄어쓰기 포함                    ... rmfjgek dlqkt rkqtl  MBCsms
  → TX4: 키 흐름 전체를 보는 Transformer, 키마다 라벨        ... k k k k k k k  e e e k k k
      판단할 단어의 앞 3단어와, 많아야 뒤 1단어를 봄
  → 스페이스: 단어 판단 (확신이 낮으면 한 단어 더 기다림)
  → 단어 안에서 언어가 바뀌는 경로면 쪼갬:                   MBC + 는
  → 후보정 (RV): 단어가 더 쳐지면 앞 단어를 바꿀 수 있음,
      확신이 높고, 단어가 늘어나도 같은 답이고, TX4가 확신이 낮았을 때만
  → 수정: Backspace → IME 전환 → raw 키 재입력
```

모든 판단은 화면 글자가 아니라 **raw 키 순서**(`dkssud`)로 합니다. 그래서 어느 모드에서 쳤든 똑같이 처리됩니다. 학습 라벨은 사람이 달 필요가 없습니다. 한글로 쓰인 글은 한국어, 알파벳으로 쓰인 글은 영어입니다.

## 결과

학습에 쓰지 않은 평가셋의 dev 절반 기준입니다. 학습 텍스트에도 있는 문장은 뺐고, test 절반은 마지막 확인용으로 손대지 않았습니다. 입력기처럼 모든 단어를 반드시 판단합니다. 스페이스에서 판단하되 확신이 99% 미만이면 한 단어 뒤에 판단하고, 단어 안에서 언어가 바뀌는 경로가 70% 이상이면 쪼갭니다. (이후 TX4에 맞춰 조정한 결과, 기다리는 기준을 99%에서 70%로 낮추면 첫 줄이 52에서 50이 되고 나중에 바뀌는 단어도 줄었습니다.)

**모델:** HMM = 키 단위 단어 태거 + 단어 단위 2상태 HMM (이전 방식). TX3, TX4 = 키 흐름 Transformer. 모든 모델과 설정값은 [models/registry.json](models/registry.json)에 있습니다.

### 틀린 단어 수

| 셋 | 단어 수 | HMM | TX3 | **TX4** |
|---|---|---|---|---|
| 실제 섞인 문장 (NSMC 리뷰 + 웹) | 15,032 | 105 | 63 | **52** |
| 실제 섞인 문장 + 오타 | 15,032 | 183 | **90** | 91 |
| 영어 + 조사 단어가 든 웹 문장 | 16,797 | 171 | 97 | **87** |
| 영어 문장의 단어 하나를 한국어로 바꾼 것 (생성) | 7,369 | 79 | 31 | **27** |
| 한국어 문장의 단어 하나를 영어로 바꾼 것 (생성) | 9,600 | 41 | 22 | **9** |
| 한국어만 | 12,819 | | **1** | 3 |
| 영어만 | 14,216 | | 4 | 4 |

오타: 단어마다 0.15 확률로 키 오타 하나(옆 키, 빠짐, 끼어듦, 순서 바뀜)를 넣었습니다. 생성 셋은 학습 데이터 일부와 같은 방식으로 만들어서 점수가 후하게 나옵니다. 믿을 건 실제 문장 셋입니다.

### 언어가 바뀌는 지점 (실제 섞인 문장)

| | 횟수 | HMM | TX3 | **TX4** |
|---|---|---|---|---|
| 한국어 속 영어로 진입 | 1,141 | 42 | 37 | **30** |
| 영어 뒤 한국어로 복귀 | 1,266 | 18 | 2 | **1** |
| 영어 + 한국어 단어를 정확히 쪼갬 (`MBC는`, 웹 셋) | 1,694 | 114 | 83 | **76** |

TX4에 남은 진입 오류 30개 중 27개가 1~3글자 토큰(`T`, `b`, `gg`)입니다.

### 단어를 나중에 고치기

TX4가 스페이스에서 판단하고, 후보정 모델이 2~5단어 뒤에 단어를 바꿀 수 있습니다. 단 충분히 확신하고, 2, 3, 5단어 뒤에 모두 같은 답이고, TX4 자신은 99% 미만으로 확신했을 때만입니다. RV0는 판단과 후보정을 같이 배우고, RV1은 후보정만 배웁니다(방금 친 단어에는 정답을 안 붙임). TX4의 스페이스 판단 대비 고침 / 망침:

| 셋 | TX4 틀림 | RV0, 확신 ≥ 90% | **RV1, 확신 ≥ 90%** | RV0, 확신 ≥ 99% | **RV1, 확신 ≥ 99%** |
|---|---|---|---|---|---|
| 키가 양쪽 언어에서 흔한 단어 (`so` / 내, `go` / 해), 정답 영어 | 1,090개 중 112 | +17 / −1 | +26 / −3 | +4 / −0 | **+9 / −0** |
| 같은 셋, 정답 한국어 | 1,017개 중 70 | +13 / −0 | +16 / −3 | +9 / −0 | +8 / −0 |
| 실제 섞인 문장 | 15,032개 중 56 | +9 / −1 | +12 / −4 | +2 / −0 | **+5 / −0** |
| 영어 + 조사 단어가 든 웹 문장 | 16,797개 중 97 | +10 / −1 | +16 / −0 | +3 / −0 | +3 / −0 |

후보정만 학습시키면 더 쓸모 있어집니다. 99% 기준에서 망침 없이 25개를 고치고(RV0는 18개), 90% 기준에서는 더 많이 고치지만 더 많이 망칩니다. 남은 오류에 비하면 아직 작은 개선입니다.

### 속도

CPU 스레드 하나로 단어당 약 17ms입니다 (파라미터 320만 개).

## 지금까지 알게 된 것

진행한 순서대로, 단계마다 무엇을 알게 됐는지입니다.

**1. 단어 단위 점수는 지나치게 낙관적이었습니다.** 첫 fastText 모델은 단어 하나 기준 오수정이 0.02%였지만 실제 문장에서는 1.03%였습니다. 짧고 흔하고 애매한 단어(그 `rm`, 내 `so`)는 단어 목록에는 드물고 실제 글에는 흔합니다. 이후로는 모든 결과를 문장으로 쟀습니다.

**2. 보정과 한 단어 대기가 문맥을 쓸 수 있게 만들었습니다.** fastText의 과신하는 점수를 누그러뜨리자 오수정이 1.03%에서 0.39%로 줄었고, 단어 단위 2상태 HMM은 한국어 문장을 0.05%까지 낮췄지만 한국어 속 영어를 한국어 쪽으로 끌어당겼습니다. 애매한 단어는 한 단어 기다리게 해서 대부분 해결했습니다.

**3. 합성 섞인 문장 셋이 진짜 문제를 가렸습니다.** 사전에서 아무 단어나 끼운 문장은 길고 드물고 쉽습니다. 실제 한국어 속 영어는 짧고 대문자인 경우가 많습니다(`TV`, `OST`, `CG`). 4상태 HMM은 합성 데이터에서 가장 좋았고 실제 데이터에서 가장 나빴습니다.

**4. 데이터를 고친 것이 어떤 문맥 기법보다 효과가 컸습니다.** 학습 단어를 실제 텍스트로 다시 만들자(대소문자 영어, 한국어 글 속 영어, 키가 겹치면 훨씬 흔한 쪽으로, 잘못된 모드로 친 한국어 제거) 문맥 없이도 실제 섞인 리뷰의 오수정이 0.68%에서 0.26%로 줄었습니다.

**5. HMM에는 넘을 수 없는 한계가 있습니다.** 단어의 확률 비율을 최대 s/(1−s)배까지만 움직일 수 있어서, 모델이 99% 확신한 단어는 절대 뒤집히지 않습니다. 남은 건 양쪽 언어에서 모두 진짜 단어인 경우(`so` / 내, `go` / 해)이고, 이걸 가르려면 옆 단어의 언어가 아니라 옆 단어 자체를 봐야 합니다.

**6. 키 단위 태거로 "단어 하나에 언어 하나" 가정을 없앴습니다.** 키마다 라벨을 붙이니까 `githubdp` → `github` + `에`는 라벨이 한 번 바뀌는 경로일 뿐입니다.

**7. 오수정은 보이는 것보다 비쌉니다.** 고친 뒤 앱이 IME 모드도 바꾸므로, 한 번 잘못 고치면 다음 단어들까지 틀린 모드로 쳐집니다.

**8. 단어 위가 아니라 키 흐름 위의 문장 모델이 낫습니다.** 단어 요약 위의 BiLSTM은 단어 태거가 알던 것(조사)을 잊었습니다. 띄어쓰기까지 포함한 키 흐름 전체를 보는 Transformer 하나는 둘 다 배웠고, 주력 모델이 됐습니다.

**9. 층을 쌓은 어텐션은 미래를 흘립니다.** 한 단어 앞까지만 보게 마스크를 걸어도, 층이 4개면 이웃을 거쳐 4단어 앞까지 닿습니다. 실제 타이핑에서는 한 단어 앞까지만 보이는데요. 그래서 학습할 때 판단하는 단어 하나(마지막 단어, 또는 마지막 단어를 다음 단어로 보는 그 앞 단어)에만 정답을 붙여, 학습과 타이핑의 조건을 똑같이 맞췄습니다.

**10. 앞 문맥은 세 단어면 충분합니다.** 앞 세 단어로 학습한 모델과 문장 전체로 학습한 모델이 같았습니다. 다음 단어를 기다려도 차이는 몇 단어 정도입니다.

**11. 데이터 필터가 정확히 어려운 경우들을 지우고 있었습니다.** 잘못된 모드로 친 한국어를 없애려고 "한국어 단어와 키가 같은 영문이 있으면 문장째 버림" 규칙을 썼더니 `so`, `go`, `do`, `work`, `B급`, 그리고 거의 모든 한 글자까지 지워졌습니다. 남은 텍스트에 한국어 속 짧은 영어가 아예 없어서 모델이 그걸 한국어로 바꿨습니다. 빈도로 판단하게 바꾸자(영어 빈도 vs 한국어 빈도, 한국어 모드에서 Shift로 못 치는 대문자는 영어) 이것들이 돌아왔습니다. 실제 섞인 문장 오류가 76에서 62로, 다음 단계까지 더해 63에서 52로 줄었습니다.

**12. 뜻이 맞는 합성 데이터가 무작위 끼우기보다 낫습니다.** 단어를 번역어로 바꾸기(`the weather` → `the 날씨`, `날씨가` → `weather가`, `미팅` / `meeting` 같은 외래어도 번역의 하나)로 무작위 끼우기를 대체했습니다. 무작위 끼우기는 같은 방식으로 만든 시험에서만 도움이 됐습니다. 한국어 글 속 영어의 30%를 소문자로 바꿔서, 사람들이 약어를 소문자로 친다는 것(`jsa`, `apk`)도 가르쳤습니다.

**13. 한 글자는 고칠 오류가 아니라 선택입니다.** 한국어 글 속의 `T`, `b`, `s`는 실제로는 한국어 자모(ㅆ, ㅠ, ㄴ)인 경우가 대부분이지만 가끔 영어입니다. "한 글자는 바꾸지 않는다" 규칙은 실제 문장에서 15개를 고쳤지만, 자모를 띄어 쓴 한국어 댓글(`ㅅ ㅂ`, `ㅁ ㅊ`)을 그만큼 망쳐서, 모델에 맡겼습니다.

**14. 나중에 고치기는 따로, 조심스럽게 해야 효과가 있습니다.** 한 모델에게 바로 판단과 나중 고치기를 같이 배우게 하자 스페이스 판단이 나빠졌습니다. 엄격한 조건을 건 별도 후보정 모델은 몇 단어를 고치고 거의 망치지 않습니다.

**15. 도움이 안 된 것들.** 학습률을 올리고 일정을 짧게 하기(확실히 나빠짐), 코드 수준 속도 개선(시간의 95%가 인코더 계산), `torch.compile`(3%), Windows 대신 WSL(11% 빠르지만 아직 옮길 만큼은 아님).

**작은 차이를 얼마나 믿을지:** dev 셋에서 모델당 오류는 수십 개 수준이고, 모든 모델을 한 번씩만 학습했습니다. 10단어 안팎의 차이는 오차 범위 안입니다.

## 재현

Windows에서 Python 3.12로 테스트했습니다. 콘솔에 한글이 출력되도록 `PYTHONUTF8=1`이 필요합니다.

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
mkdir -p data/raw/kcbert data/raw/mc4 data/raw/wiktionary
curl -L -o data/raw/nsmc_train.txt https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt
curl -L -o data/raw/nsmc_test.txt https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt
curl -L -o data/raw/eng_sentences.tsv.bz2 https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2
curl -L -o data/raw/kor_sentences.tsv.bz2 https://downloads.tatoeba.org/exports/per_language/kor/kor_sentences.tsv.bz2
for p in aa ab ac; do curl -L -o data/raw/kcbert/kcbert-train.tar.gz$p https://github.com/Beomi/KcBERT/releases/download/TrainData_v1/kcbert-train.tar.gz$p; done
curl -L -o data/raw/mc4/c4-ko-validation.json.gz https://huggingface.co/datasets/allenai/c4/resolve/main/multilingual/c4-ko-validation.tfrecord-00000-of-00001.json.gz
curl -L -o data/raw/wiktionary/kaikki-korean.jsonl https://kaikki.org/dictionary/Korean/kaikki.org-dictionary-Korean.jsonl
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

모델은 PyTorch로 학습합니다. Windows의 AMD GPU는 AMD의 ROCm 패키지를 별도 venv에 설치합니다([AMD 안내](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)). CPU라면 `pip install torch`로 충분합니다. TX4 (Radeon RX 9060 XT에서 약 25분):

```bash
NUNCHI_KO=0.25 NUNCHI_EN=0.20 NUNCHI_WORDS=0.15 NUNCHI_KMIX=0.113 NUNCHI_WEB=0.187 \
NUNCHI_TRANS_KO=0.05 NUNCHI_TRANS_EN=0.05 NUNCHI_SYNTH=0 NUNCHI_LOWER_P=0.3 \
NUNCHI_ONE_WORD=1 NUNCHI_LEFT=3 NUNCHI_SAVE_LAST=1 NUNCHI_OUT=models/tx4.pt \
.venv-rocm/Scripts/python -m nunchi.stream_tagger train tx4_log.txt 17000 2125 1000 1024 3
```

중간에 멈춘 학습은 `NUNCHI_RESUME=1`로 체크포인트에서 이어갑니다. 위 표를 만든 벤치마크 스크립트는 아직 저장소에 없습니다.

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
| `nunchi/stream_tagger.py` | 키 흐름 Transformer (TX, RV), 이어하기 되는 학습, `decide()` |
| `nunchi/models.py` | `models/registry.json`에서 이름으로 모델과 문맥, 기준값 불러오기 |
| `nunchi/eval_sentence.py` | 문장의 모든 단어 판단 (바로 / 한 단어 뒤), 단어 쪼개기 |
| `nunchi/export_onnx.py` | 앱용 ONNX 변환 |
| `nunchi/context.py` | HMM 문맥 실험, 실제 섞인 리뷰를 포함한 문장 셋 |
| `nunchi/demo.py`, `typebench.py` | 로컬 테스트 페이지, 실제 키 입력 벤치마크 |
| `experiments/` | 숫자를 살린 평가셋, 종류별 표, 판단 기준 곡선, 설정 파일로 학습을 돌리는 `run.py` |
| `docs/` | 실험 기록과 평가 가이드 |
| `app/` | Windows 앱 (만드는 중) |

## 로드맵

- [x] 한글 → 두벌식 키, fastText 기준선, HMM 문맥
- [x] 실제 섞인 문장 평가셋 (NSMC 리뷰, mC4 웹 글), 학습과 분리, 중복 제거
- [x] 키 단위 라벨링 모델 (단어 안 전환: `MBC는`, `github에`)
- [x] 키 흐름 위의 문장 모델 (TX4), 타이핑과 같은 조건의 학습
- [x] 데이터: 빈도 기반 잘못된 모드 필터, mC4, 번역 치환, 소문자 증강
- [x] ONNX 변환
- [x] 후보정: 전용 모델(RV1)
- [ ] 바로 판단과 후보정을 TX4 + RV1만큼 하는 한 모델
- [ ] 실제 타이핑 로그로 만든 평가셋 (위 평가셋은 리뷰와 웹 글)
- [ ] 같은 설정 재학습의 흔들림 측정 (작은 차이를 판단하기 위해)
- [ ] 더 길게 학습 (17,000스텝에서도 손실이 내려가는 중이었음)
- [ ] 벤치마크 스크립트를 저장소에
- [ ] Windows 앱: 키보드 훅, IME 제어, 예외 처리, 되돌리기, 후보정

## 데이터 출처

- [NSMC](https://github.com/e9t/nsmc) (네이버 영화 리뷰): 한국어, CC0
- [KcBERT 사전학습 말뭉치](https://github.com/Beomi/KcBERT) (뉴스 댓글): 한국어, CC BY-SA 4.0
- [mC4](https://huggingface.co/datasets/allenai/c4) 한국어 (Common Crawl 웹 글): ODC-BY, 그리고 Common Crawl 이용 약관
- [Tatoeba](https://tatoeba.org): 영어, 한국어 문장, CC BY 2.0 FR
- [위키낱말사전](https://en.wiktionary.org) 한국어 항목, [kaikki.org](https://kaikki.org) 경유 (Tatu Ylonen, "Wiktextract: Wiktionary as Machine-Readable Structured Data", LREC 2022): CC BY-SA 4.0 / GFDL
- [wordfreq](https://github.com/rspeer/wordfreq): 영어 단어와 빈도, 데이터는 CC BY-SA 4.0

이 저장소에는 위 출처에서 나온 데이터가 없습니다. 위 스크립트로 다시 만듭니다. 학습된 가중치도 저장소에 없습니다. CC BY-SA 데이터로 학습했으므로, 가중치를 공개한다면 CC BY-SA 4.0을 따릅니다.
