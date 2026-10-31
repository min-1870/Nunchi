# Nunchi (gksdud)

[English](README.md) | **한국어**

> 한/영 키 안 눌러도 알아서 고쳐주는 Windows 입력 도우미

영어 모드에서 `dkssud`를 치면 `안녕`으로, 한글 모드에서 `ㅗ디ㅣㅐ`를 치면 `hello`로 바꿔줍니다. 입력 모드도 같이 바뀝니다.
저장소 이름 `gksdud`는 영어 모드에서 "한영"을 친 모양입니다.

> **상태: 연구 단계.** 판별기와 평가는 동작합니다. 키 단위 태거는 영어에 조사가 붙은 단어(`MBC는`)까지 처리합니다. 문장 단위 모델은 학습 중입니다. Windows 트레이 앱은 아직 없습니다.

## 원칙

- **100% 로컬.** 네트워크를 전혀 쓰지 않습니다. 키 입력을 다루는 프로그램이라 신뢰가 먼저입니다.
- **확신 없으면 안 건드림.** 잘못 고치는 것이 못 고치는 것보다 비쌉니다.
- **되돌리기 단축키.** 되돌린 단어는 예외로 학습해서 다시는 고치지 않습니다.
- **단어 경계 후 100ms 안에 수정.**

## 동작 방식

```
키 입력 수집 (WH_KEYBOARD_LL)
  → 단어 경계까지 raw 키를 버퍼링
  → 키 태거: 키마다 언어 라벨 (BiLSTM-CRF)
      dkssud   → k k k k k k      → 안녕
      MBCsms   → e e e k k k      → MBC는
  → 문맥: 단어 단위 2상태 HMM (문장 모델로 교체 중)
  → 지금 확실함?        바로 수정
    애매함?             한 단어 기다렸다가 소급 수정
    그래도 0.99 미만?   아무것도 안 함
  → 수정: Backspace → IME 전환 → raw 키 재입력
```

모든 판단은 화면 글자가 아니라 **raw 키 순서**(`dkssud`)로 합니다. 그래서 어느 모드에서 쳤든 똑같이 처리됩니다. 학습 라벨은 사람이 달 필요가 없습니다. 한글로 쓰인 글은 한국어, 알파벳으로 쓰인 글은 영어입니다.

## 결과

모든 숫자는 학습에 쓰지 않은 데이터 기준입니다. test 단어는 학습에 나오지 않고(단어 종류 단위로 분할), 문장 셋은 모델이 본 적 없는 글에서 가져왔습니다.

### 문장, 모든 단어를 판정할 때

입력기처럼 모든 단어를 반드시 판정해야 할 때, 틀린 언어로 나온 단어 수입니다. 문맥은 2상태 HMM이고, 바로 고치거나 애매하면 한 단어 기다립니다.

| 셋 | 단어 수 | fastText (v3) | **키 태거** |
|---|---|---|---|
| 한국어 문장 (NSMC 리뷰) | 12,819 | 14 | **6** |
| 영어 문장 (Tatoeba) | 14,216 | **0** | 2 |
| 한국어 + 영어 끼우기 (합성) | 16,154 | 49 | **48** |
| 영어 + 한국어 끼우기 (합성) | 17,565 | 101 | **57** |
| **실제 섞인 리뷰** (NSMC test) | 8,599 | 88 | **82** |
| 한국어 문장 + 오타 | 12,819 | 98 | **36** |
| 영어 문장 + 오타 | 14,216 | 55 | **28** |
| 실제 섞인 리뷰 + 오타 | 8,599 | 149 | **109** |

오타: 단어마다 0.15 확률로 키 오타 하나(옆 키, 빠짐, 끼어듦, 순서 바뀜)를 넣었습니다.

### 문장, 확신할 때만 고칠 때 (≥ 0.99)

"오수정"은 고친 것 중 틀린 비율, "고친 비율"은 모델이 확신해서 손댄 단어의 비율입니다.

| 셋 | fastText v3: 오수정 / 고친 비율 | **키 태거: 오수정 / 고친 비율** |
|---|---|---|
| 한국어 문장 | 0.05% / 99.0% | **0.00% / 98.0%** |
| 영어 문장 | 0% / 98.7% | 0% / 97.6% |
| 한국어 + 영어 끼우기 | 0.08% / 98.7% | **0.01% / 97.4%** |
| 영어 + 한국어 끼우기 | 0.05% / 98.0% | **0.00% / 95.3%** |
| **실제 섞인 리뷰** | 0.48% / 97.9% | **0.17% / 95.6%** |

### 단어 하나

| 같은 test 단어, 모든 단어 판정 | fastText v3 | **키 태거** |
|---|---|---|
| 한국어 단어 | 99.73% | **99.75%** |
| 영어 단어 | **99.26%** | 99.09% |
| 한국어 단어 + 오타 하나 | 97.53% | **98.64%** |
| 영어 단어 + 오타 하나 | **98.23%** | 97.83% |
| 한 단어 안의 영어 + 한국어 조사 (`MBC는`, `TV에서`), 실제 토큰 | 불가능 | **90.4%** |

조사 테스트는 말뭉치의 실제 토큰 중 영어 부분이 학습에 한 번도 나오지 않은 것들입니다.

## 지금까지 알게 된 것

진행한 순서대로, 단계마다 무엇을 알게 됐는지입니다.

**1. 단어 단위 점수는 지나치게 낙관적이었습니다.** 첫 fastText 모델은 단어 하나 기준 오수정이 0.02%였지만, 실제 문장에서는 1.03%였습니다. 짧고 흔하고 애매한 단어(그 `rm`, 내 `so`)는 단어 목록에는 드물고 실제 글에는 흔합니다. 이후로는 모든 결과를 문장으로 쟀습니다.

**2. 보정과 한 단어 대기가 문맥을 쓸 수 있게 만들었습니다.** fastText는 틀려도 p = 1.000이라고 합니다. 점수를 누그러뜨리는 것(temperature)만으로 오수정이 1.03%에서 0.39%로 줄었습니다. 단어 단위 2상태 HMM(앞 단어와 같은 언어일 확률)은 한국어 문장을 0.05%까지 낮췄지만, 한국어 문장에 처음 끼어든 영어를 한국어 쪽으로 끌어당겼습니다. 애매한 단어는 한 단어 기다렸다 판단하게 해서 대부분 해결했습니다.

**3. 합성 섞인 문장 셋이 진짜 문제를 가렸습니다.** 사전에서 아무 단어나 뽑아 끼워서(`disgruntled`, `hierarchical`) 길고 드물고 쉬웠습니다. 실제 리뷰 속 영어는 짧고 대문자입니다(`TV`, `OST`, `CG`). 첫 모델은 이런 걸 학습에서 본 적이 없었습니다(wordfreq는 전부 소문자). 문장의 기본 언어를 기억하는 4상태 HMM은 합성 데이터에서는 가장 좋았고 실제 데이터에서는 가장 나빴습니다. 그 짧은 영어를 한국어로 끌어당겨서, 확신하고 고친 것의 50%가 틀렸습니다.

**4. 데이터를 고친 것이 어떤 문맥 기법보다 효과가 컸습니다.** 학습 단어를 실제 텍스트로 다시 만들었더니, 문맥 없이도 실제 섞인 리뷰의 오수정이 0.68%에서 0.26%로 줄었습니다.
- 사람이 실제로 치는 영어: 실제 문장의 대소문자 형태, 한국어 글 속에 나온 영어.
- 버그: 영어 단어 목록에 같은 키의 잡음 단어가 있어서, 흔한 한국어 단어(이 `dl`, 수 `tn`, 그 `rm`, ㅋㅋ `zz`)가 학습에서 지워져 있었습니다. 이제 키가 겹치면 훨씬 흔한 쪽으로 보내고, 비슷하게 흔하면(`so` / 내) 지우지 않고 양쪽에 다 남깁니다.
- 리뷰 속에 모드를 잘못 놓고 친 한국어(`gks` = 한)가 영어로 학습돼 있었습니다.
- 영화 리뷰 말고도 뉴스 댓글(KcBERT)과 Tatoeba 한국어를 넣어서, 일상 단어(키, 메뉴)를 알게 됐습니다.

**5. 문맥에는 넘을 수 없는 한계가 있고, 그걸 재봤습니다.**
- HMM은 뒤에 단어를 몇 개 보든, 한 단어의 확률 비율을 최대 s/(1−s)배까지만 움직일 수 있습니다. 모델이 99% 확신한 단어는 뒤 단어로 절대 뒤집히지 않습니다. 그래서 "고쳤다가 다음 단어 보고 되돌리기"는 모든 셋에서 한 번도 일어나지 않았습니다.
- 옆 단어의 진짜 언어를 알려주는 오라클로 재보면, 문맥은 한 언어 문장의 못 고친 단어는 대부분 없애지만 언어가 바뀌는 지점에서는 하나도 못 없앱니다.
- 그래서 HMM은 주로 고치는 단어 수를 늘리고, 오수정은 거의 못 줄입니다. 남은 건 양쪽 언어에서 모두 진짜 단어인 경우입니다(`so` / 내, `go` / 해, `do` / 양). 이걸 가르려면 옆 단어의 언어가 아니라 옆 **단어 자체**를 봐야 합니다. 문장 모델이 그 역할입니다.

**6. 키 단위 태거로 "단어 하나에 언어 하나" 가정을 없앴습니다.** 키마다 라벨을 붙이니까 `githubdp` → `github` + `에`는 라벨이 한 번 바뀌는 경로일 뿐입니다. 단어 전체 판정은 fastText와 비슷하고, 한국어 오타에는 더 강하고, 영어 + 조사(실제 처음 보는 토큰 90.4%)를 처리합니다. 단어 분류기로는 아예 못 하는 일입니다.

**7. 오수정은 보이는 것보다 비쌉니다.** 고칠 때 앱이 IME 모드까지 바꾸므로, 잘못 고치면 그다음 단어들까지 틀린 모드로 쳐집니다. 반대로 못 고친 단어는 사용자가 실제로 잘못된 모드에 있을 때만 의미가 있습니다. 치기 시작할 때, 창을 바꾼 직후, 언어를 바꾸면서 한/영을 깜빡했을 때입니다. 이곳들은 문맥이 가장 도움이 안 되는 곳이라, 단어 단위 모델이 가장 중요합니다.

## 재현

Windows에서 Python 3.12로 테스트했습니다. 콘솔에 한글이 출력되도록 `PYTHONUTF8=1`이 필요합니다.

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
mkdir -p data/raw/kcbert
curl -L -o data/raw/nsmc_train.txt https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt
curl -L -o data/raw/nsmc_test.txt https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt
curl -L -o data/raw/eng_sentences.tsv.bz2 https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2
curl -L -o data/raw/kor_sentences.tsv.bz2 https://downloads.tatoeba.org/exports/per_language/kor/kor_sentences.tsv.bz2
for p in aa ab ac; do curl -L -o data/raw/kcbert/kcbert-train.tar.gz$p https://github.com/Beomi/KcBERT/releases/download/TrainData_v1/kcbert-train.tar.gz$p; done
export PYTHONUTF8=1
.venv/Scripts/python -m nunchi.count_corpus     # 출처별 단어 / 영어 / 영어+한글 개수
.venv/Scripts/python -m nunchi.make_data        # 단어 train/valid/test
.venv/Scripts/python -m nunchi.make_noise       # 오타/코드 test셋, 오타 학습 데이터
.venv/Scripts/python -m nunchi.compare          # fastText 모델 (ft_shape_aug.ftz 학습) vs 규칙
.venv/Scripts/python -m nunchi.context          # 문장 문맥 (fastText)
```

키 태거는 PyTorch로 학습합니다. Windows의 AMD GPU는 AMD의 ROCm 패키지를 별도 venv에 설치합니다([AMD 안내](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)). CPU라면 `pip install torch`로 충분합니다.

```bash
.venv-rocm/Scripts/python -m nunchi.tagger train
.venv-rocm/Scripts/python -m nunchi.tagger eval
.venv-rocm/Scripts/python -m nunchi.context tagger
.venv/Scripts/python -m nunchi.demo             # 직접 쳐보기: http://localhost:8765
```

## 구성

| 파일 | 역할 |
|---|---|
| `nunchi/dubeolsik.py` | 한글 → 두벌식 키 변환, 한글 모드로 읽었을 때의 음절 모양 |
| `nunchi/count_corpus.py` | 출처별 개수: 한국어 단어, 한국어 글 속 영어, 영어 + 조사 |
| `nunchi/make_data.py` | 단어 데이터: 말뭉치의 한국어, wordfreq + 실제 대소문자 영어 |
| `nunchi/make_noise.py` | 오타/코드 토큰 test셋, 오타 학습 데이터 |
| `nunchi/train_fasttext.py` | fastText 학습, 양자화, 빈도 가중 평가 |
| `nunchi/eval_rules.py`, `compare.py` | 규칙 기준선, 단어 단위 규칙 vs 모델 vs 혼합 |
| `nunchi/tagger.py` | 키 단위 BiLSTM-CRF 태거 (한 단어 안의 영어 + 조사) |
| `nunchi/context.py` | 문장 셋(실제 섞인 리뷰 포함), 보정, HMM 문맥, 한 단어 대기 |
| `nunchi/demo.py` | 로컬 테스트 페이지 |
| `nunchi/typebench.py` | 메모장에 실제 키 입력을 보내는 벤치마크 |

## 로드맵

- [x] 한글 → 두벌식 키 변환
- [x] fastText 기준선 + 평가 리포트
- [x] 문맥: 보정, 이전 단어 HMM, 한 단어 대기
- [x] 실제 섞인 문장 test셋 (이미 영어가 섞인 NSMC 리뷰)
- [x] 실제 텍스트로 학습 데이터 재구성 (대소문자 영어, 넓힌 한국어, 키 충돌 처리)
- [x] 키 단위 라벨링 모델 (단어 안 전환: `MBC는`, `github에`)
- [ ] 문장 단위 문맥 모델 (옆 단어의 언어가 아니라 옆 단어 자체) — 학습 중
- [ ] 실사용 시뮬레이션: 잘못된 모드로 시작, 한/영 깜빡, 오수정 뒤 연쇄
- [ ] ONNX 변환
- [ ] 한글 모드 출력 글자 수 계산 (Backspace용: 종성 이동, 겹받침) + 왕복 테스트
- [ ] C# 트레이 앱: 키보드 훅, IME 제어, 예외(비밀번호 칸, 게임, 관리자 창), 되돌리기, 소급 수정
- [ ] 개발자 채팅 / 코드 주석에서 만든 섞인 문장 test셋

## 데이터 출처

- [NSMC](https://github.com/e9t/nsmc) (네이버 영화 리뷰): 한국어, CC0
- [KcBERT 사전학습 말뭉치](https://github.com/Beomi/KcBERT) (뉴스 댓글): 한국어, CC BY-SA 4.0. 학습에만 쓰고, 이 저장소에는 여기서 나온 데이터가 없습니다
- [Tatoeba](https://tatoeba.org): 영어, 한국어 문장, CC BY 2.0 FR
- [wordfreq](https://github.com/rspeer/wordfreq): 영어 단어와 빈도, 데이터는 CC BY-SA 4.0
