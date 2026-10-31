# Nunchi (gksdud)

[English](README.md) | **한국어**

> 한/영 키 안 눌러도 알아서 고쳐주는 Windows 입력 도우미

영어 모드에서 `dkssud`를 치면 `안녕`으로, 한글 모드에서 `ㅗ디ㅣㅐ`를 치면 `hello`로 고치고 IME 모드도 같이 바꿔줍니다.
저장소 이름 `gksdud`은 영어 모드에서 친 "한영"입니다.

> **현재 상태: 초기 연구 단계.** 판별 모델(fastText + 문맥)과 평가까지 끝났고, Windows 트레이 앱은 아직 없습니다.

## 원칙

- **100% 로컬 처리.** 네트워크 통신이 없습니다. 키 입력을 다루는 프로그램이라 신뢰가 최우선입니다.
- **확신이 없으면 아무것도 하지 않습니다.** 잘못 고치는 비용이 안 고치는 비용보다 큽니다.
- **되돌리기 핫키를 제공합니다.** 되돌린 단어는 다시 고치지 않도록 학습합니다.
- **단어 경계 후 100ms 이내에 수정합니다.**

## 동작 방식 (계획)

```
키 입력 캡처 (WH_KEYBOARD_LL)
  → 단어 경계까지 raw 키 버퍼링
  → 판별:
      1. 확정 규칙 (비밀번호 필드, 게임, 되돌린 단어)
      2. 단어 테이블 (대소문자 구분: 한글 모드에서 Shift는 의미가 있음)
      3. fastText 점수 + 보정 (temperature)
      4. 문맥: 앞 단어들을 보는 2상태 HMM
  → 지금 확실하면           바로 수정
    애매하면                다음 단어까지 보고 소급 수정
    그래도 0.99 미만이면     아무것도 안 함
  → 수정: Backspace → IME 전환 → raw 키 재입력
```

판별은 화면에 찍힌 글자가 아니라 **raw 키 시퀀스**(`dkssud`)를 보고 합니다. 그래서 어느 모드에서 쳤든 같은 입력으로 다룰 수 있습니다.

## 결과

### 실제 문장 기준

한국어 영화 리뷰(NSMC), 영어 문장(Tatoeba), 한국어 문장 중간에 영어 단어를 끼워 넣은 섞인 문장으로 평가했습니다.
단어가 자연스러운 빈도로 나오기 때문에, 그(`rm`), 내(`so`), 지(`wl`)처럼 짧고 자주 쓰는 단어가 실제 타이핑만큼 자주 나옵니다.

"오수정"은 확신도 0.99 이상일 때만 고친다고 가정했을 때, 고친 것 중 틀린 비율입니다.

| | 문맥 없음 | 문맥 없음 + 보정 | **문맥: 바로 or 한 단어 대기** |
|---|---|---|---|
| 한국어 문장 | 오수정 1.03% | 오수정 0.39% | **오수정 0.05%** |
| 영어 문장 | 오수정 0% | 오수정 0% | 오수정 0% |
| 섞인 문장 | 오수정 0.76% | 오수정 0.32% | **오수정 0.23%** |
| 언어 전환 지점: 고친 비율 / 오수정 | 99% / 0.65% | 95% / 0.45% | 63% / 0.95% |

- **문맥을 쓰면 한국어 오수정이 약 1/8로 줄어듭니다.** 단어 하나로는 애매한 짧은 단어(그, 내, ㅜㅜ)가 주변 단어 덕분에 풀립니다.
- **문맥을 쓰려면 다음 단어까지 봐야 합니다.** 앞 단어만 보면 한국어 문장 속 첫 영어 단어가 한국어 쪽으로 끌려갑니다. 그 경우 27%만 고쳤고, 고친 것 중 2.7%가 오수정이었습니다. 다음 단어까지 보면("오늘 fork 했는데") 63% / 0.95%가 됩니다. 한국어 문장에서 대기가 필요한 단어는 4%뿐입니다.
- **보정이 중요합니다.** fastText는 틀릴 때도 p=1.000이라고 합니다. 점수를 부드럽게 펴는 것(temperature T=2)만으로 오수정이 1.03%에서 0.39%로 줄어듭니다.
- **언어 전환 지점은 아직 약점입니다.** 남은 오류의 절반이 영어 단어 바로 뒤의 `zz`(ㅋㅋ)입니다. ㅋㅋ류 테이블 규칙으로 해결될 것으로 봅니다.

### 단어 단위: 규칙 vs 모델

test 단어는 하나씩 따로 판별하고, 학습 때 한 번도 보지 않은 단어입니다(단어 종류 단위로 분리). 자주 치는 단어일수록 큰 가중치를 줬습니다.

| 방식 | 크기 | 깨끗한 단어 | 오타 섞인 단어 | 코드 토큰 |
|---|---|---|---|---|
| 규칙: 완성 음절이면 한국어 | 0 | 오수정 3.66% | 오수정 26.8% | 99.0% |
| 규칙: 잘 만든 버전 (사전 + 테이블) | 0 | 오수정 0.09% | 오수정 25.4% | 99.0% |
| fastText (오타 학습) | 1.5MB | 오수정 0.02% | 오수정 0.57% | 99.1% |
| **하이브리드 (테이블 + fastText)** | 1.5MB | **오수정 0.02%** | **오수정 0.52%** | 99.1% |

코드 토큰 열은 정확도입니다(`getUser`, `user_id`).

- **깨끗하고 자주 쓰는 단어는 규칙만으로도 충분합니다.** 잘 만든 규칙의 빈도 가중 정확도는 99.91%로, 모델과 같습니다.
- **ML의 가치는 오타 처리입니다.** 오타 하나로 음절이 깨지면 규칙은 한국어를 영어로 판정합니다(`wpoalTdjdy`는 재밌어요에 오타 하나). 오타 섞인 입력에서 모델의 오수정은 규칙의 약 1/50입니다.
- **Shift는 정보입니다.** 테이블 조회에서 대소문자를 무시하면 `Tl`(씨)이 영어 `tl`로 판정됩니다.
- **양자화하면 37MB가 1.4MB로 줄고**, 정확도는 그대로입니다.
- **이 단어 단위 숫자는 낙관적입니다.** 여기서 0.02%인 오수정이 실제 문장에서는 0.39%가 됩니다(보정, 문맥 없음 기준). 짧고 자주 쓰는 애매한 단어가 단어 단위 test에서 덜 반영됐기 때문입니다.

## 재현

Python 3.12, Windows 기준입니다. 콘솔 한글 출력 때문에 `PYTHONUTF8=1`이 필요합니다.

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
mkdir -p data/raw
curl -L -o data/raw/nsmc_train.txt https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt
curl -L -o data/raw/nsmc_test.txt https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt
curl -L -o data/raw/eng_sentences.tsv.bz2 https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2
export PYTHONUTF8=1
.venv/Scripts/python -m nunchi.make_data        # train/valid/test 생성
.venv/Scripts/python -m nunchi.train_fasttext   # 모델 학습 + 리포트 (6스레드)
.venv/Scripts/python -m nunchi.make_noise       # 오타/코드 test셋, 오타 학습 데이터
.venv/Scripts/python -m nunchi.compare          # 규칙 vs 모델 vs 하이브리드 (단어 단위)
.venv/Scripts/python -m nunchi.context          # 실제 문장에서 문맥 평가
```

## 구조

| 파일 | 역할 |
|---|---|
| `nunchi/dubeolsik.py` | 한글 → 두벌식 키 변환, 한글 모드에서의 음절 형태 판정 |
| `nunchi/make_data.py` | 합성 데이터 생성 (NSMC 어절 + wordfreq 영어 단어) |
| `nunchi/make_noise.py` | 오타/코드 토큰 test셋, 오타 학습 데이터 |
| `nunchi/train_fasttext.py` | fastText 학습, 양자화, 빈도 가중 평가 |
| `nunchi/eval_rules.py` | 규칙만 쓰는 baseline |
| `nunchi/compare.py` | 단어 단위 규칙 / 모델 / 하이브리드 비교 |
| `nunchi/context.py` | 문장 단위 보정 + HMM 문맥 + 한 단어 대기 |

## 로드맵

- [x] 두벌식 역변환
- [x] 합성 데이터 생성기
- [x] fastText baseline + 평가 리포트
- [x] 문맥: 보정, 앞 단어 HMM, 한 단어 대기
- [ ] PyTorch 재구현 → ONNX
- [ ] 한글 모드 출력 글자 수 계산 (Backspace용, 종성 이동·겹받침) + 왕복 테스트
- [ ] C# 트레이 앱: 키보드 훅, IME 제어, 제외 규칙(비밀번호 필드, 게임, 관리자 권한 창), 되돌리기, 소급 수정
- [ ] 키 단위 태깅 모델 (단어 내부 전환: `API를`, `React에서`)
- [ ] 실제에 가까운 한영 혼합 test셋 (개발자 채팅, 코드 주석)

## 데이터 출처

- [NSMC](https://github.com/e9t/nsmc) (네이버 영화 리뷰): 한국어 텍스트, CC0
- [wordfreq](https://github.com/rspeer/wordfreq): 영어 단어와 빈도, 데이터는 CC BY-SA 4.0
- [Tatoeba](https://tatoeba.org): 영어 문장, CC BY 2.0 FR
