"""Kiwi 형태소 분석기 기반 BM25 Retriever.
References: https://github.com/bab2min/kiwipiepy
"""
from __future__ import annotations

import asyncio
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

try:
    from kiwipiepy import Kiwi, NgramExtractor
    from kiwipiepy.utils import Stopwords
except ImportError:
    print("Could not import kiwipiepy, please install with `pip install kiwipiepy`.")
    raise

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Could not import rank_bm25, please install with `pip install rank-bm25`.")
    raise


# ============================================================================
# 세종 품사 태그 상수 및 편의 함수
# ============================================================================

# 세종 품사 태그 체계
# 대분류별 품사 태그 정의
SEJONG_POS_TAGS = {
    # 체언 (Noun)
    "NOUN": {
        "NNG": "일반 명사",
        "NNP": "고유 명사",
        "NNB": "의존 명사",
        "NR": "수사",
        "NP": "대명사",
    },
    # 용언 (Verb/Adjective)
    "PREDICATE": {
        "VV": "동사",
        "VA": "형용사",
        "VX": "보조 용언",
        "VCP": "긍정 지시사(이다)",
        "VCN": "부정 지시사(아니다)",
    },
    # 관형사
    "DETERMINER": {
        "MM": "관형사",
    },
    # 부사
    "ADVERB": {
        "MAG": "일반 부사",
        "MAJ": "접속 부사",
    },
    # 감탄사
    "INTERJECTION": {
        "IC": "감탄사",
    },
    # 조사
    "PARTICLE": {
        "JKS": "주격 조사",
        "JKC": "보격 조사",
        "JKG": "관형격 조사",
        "JKO": "목적격 조사",
        "JKB": "부사격 조사",
        "JKV": "호격 조사",
        "JKQ": "인용격 조사",
        "JX": "보조사",
        "JC": "접속 조사",
    },
    # 어미
    "ENDING": {
        "EP": "선어말 어미",
        "EF": "종결 어미",
        "EC": "연결 어미",
        "ETN": "명사형 전성 어미",
        "ETM": "관형형 전성 어미",
    },
    # 접두사
    "PREFIX": {
        "XPN": "체언 접두사",
    },
    # 접미사
    "SUFFIX": {
        "XSN": "명사 파생 접미사",
        "XSV": "동사 파생 접미사",
        "XSA": "형용사 파생 접미사",
        "XSM": "부사 파생 접미사",  # Kiwi 독자적 태그
    },
    # 어근
    "ROOT": {
        "XR": "어근",
    },
    # 부호, 외국어, 특수문자
    "SYMBOL": {
        "SF": "종결 부호(. ! ?)",
        "SP": "구분 부호(, / : ;)",
        "SS": "인용 부호 및 괄호",
        "SSO": "SS 중 여는 부호",  # Kiwi 독자적 태그
        "SSC": "SS 중 닫는 부호",  # Kiwi 독자적 태그
        "SE": "줄임표(…)",
        "SO": "붙임표(- ~)",
        "SW": "기타 특수 문자",
        "SL": "알파벳(A-Z a-z)",
        "SH": "한자",
        "SN": "숫자(0-9)",
        "SB": "순서 있는 글머리",  # Kiwi 독자적 태그
    },
    # 분석 불능
    "UNKNOWN": {
        "UN": "분석 불능",  # Kiwi 독자적 태그
    },
    # 웹 관련
    "WEB": {
        "W_URL": "URL 주소",  # Kiwi 독자적 태그
        "W_EMAIL": "이메일 주소",  # Kiwi 독자적 태그
        "W_HASHTAG": "해시태그(#abcd)",  # Kiwi 독자적 태그
        "W_MENTION": "멘션(@abcd)",  # Kiwi 독자적 태그
        "W_SERIAL": "일련번호(전화번호, 통장번호, IP주소 등)",  # Kiwi 독자적 태그
        "W_EMOJI": "이모지",  # Kiwi 독자적 태그
    },
    # 기타
    "OTHER": {
        "Z_CODA": "덧붙은 받침",  # Kiwi 독자적 태그
        "Z_SIOT": "사이시옷",  # Kiwi 독자적 태그
        "USER0": "사용자 정의 태그 0",  # Kiwi 독자적 태그
        "USER1": "사용자 정의 태그 1",  # Kiwi 독자적 태그
        "USER2": "사용자 정의 태그 2",  # Kiwi 독자적 태그
        "USER3": "사용자 정의 태그 3",  # Kiwi 독자적 태그
        "USER4": "사용자 정의 태그 4",  # Kiwi 독자적 태그
    },
}

# 모든 품사 태그 집합
ALL_POS_TAGS: set[str] = set()
for category in SEJONG_POS_TAGS.values():
    ALL_POS_TAGS.update(category.keys())

# 품사 태그 설명 딕셔너리 (태그 -> 설명)
POS_TAG_DESCRIPTIONS: dict[str, str] = {}
for category in SEJONG_POS_TAGS.values():
    POS_TAG_DESCRIPTIONS.update(category)


def get_pos_tags_by_category(category: str) -> set[str]:
    """대분류에 속한 모든 품사 태그를 반환한다.

    매개변수:
        category: 대분류 이름 ('NOUN', 'PREDICATE', 'PARTICLE' 등)

    반환값:
        해당 대분류에 속한 품사 태그 집합

    예시:
        >>> get_pos_tags_by_category("NOUN")
        {'NNG', 'NNP', 'NNB', 'NR', 'NP'}
        >>> get_pos_tags_by_category("PREDICATE")
        {'VV', 'VA', 'VX', 'VCP', 'VCN'}
    """
    return set(SEJONG_POS_TAGS.get(category, {}).keys())


def get_pos_description(tag: str) -> str:
    """품사 태그의 설명을 반환한다.

    매개변수:
        tag: 품사 태그 (예: 'NNG', 'VV')

    반환값:
        품사 태그 설명

    예시:
        >>> get_pos_description("NNG")
        '일반 명사'
        >>> get_pos_description("VV")
        '동사'
    """
    return POS_TAG_DESCRIPTIONS.get(tag, f"알 수 없는 품사 태그: {tag}")


def get_default_important_pos() -> set[str]:
    """BM25 검색에 기본적으로 사용할 품사 집합을 반환한다.

    명사, 동사, 형용사, 영어, 한자, 숫자 등 의미 있는 단어만 포함한다.

    반환값:
        기본 품사 태그 집합

    예시:
        >>> pos_set = get_default_important_pos()
        >>> "NNG" in pos_set
        True
        >>> "JKS" in pos_set  # 조사는 제외
        False
    """
    return {
        "NNG",  # 일반 명사
        "NNP",  # 고유 명사
        "NNB",  # 의존 명사
        "VV",  # 동사
        "VA",  # 형용사
        "SL",  # 알파벳
        "SH",  # 한자
        "SN",  # 숫자
    }


def get_noun_only_pos() -> set[str]:
    """명사만 포함하는 품사 집합을 반환한다.

    반환값:
        명사 품사 태그 집합

    예시:
        >>> pos_set = get_noun_only_pos()
        >>> pos_set == {"NNG", "NNP", "NNB", "NR", "NP"}
        True
    """
    return get_pos_tags_by_category("NOUN")


def get_content_word_pos() -> set[str]:
    """내용어(명사, 동사, 형용사)만 포함하는 품사 집합을 반환한다.

    반환값:
        내용어 품사 태그 집합

    예시:
        >>> pos_set = get_content_word_pos()
        >>> "NNG" in pos_set and "VV" in pos_set and "VA" in pos_set
        True
    """
    return get_pos_tags_by_category("NOUN") | get_pos_tags_by_category("PREDICATE")


def get_all_content_pos() -> set[str]:
    """모든 내용어(명사, 동사, 형용사, 부사, 관형사)를 포함하는 품사 집합을 반환한다.

    반환값:
        모든 내용어 품사 태그 집합
    """
    return (
        get_pos_tags_by_category("NOUN")
        | get_pos_tags_by_category("PREDICATE")
        | get_pos_tags_by_category("ADVERB")
        | get_pos_tags_by_category("DETERMINER")
    )


# Kiwi 싱글톤 캐시 (모델별)
_KIWI_INSTANCES: dict[str, Kiwi] = {}


def get_kiwi_instance(
    model_type: str = "knlm",
    typos: str | None = None,
    model_path: str | None = None,
    load_default_dict: bool = True,
    load_typo_dict: bool = True,
    load_multi_dict: bool = True,
    integrate_allomorph: bool = True,
    typo_cost_threshold: float = 2.5,
    num_workers: int = -1,
) -> Kiwi:
    """Kiwi 인스턴스를 캐시하여 반환한다.

    동일한 설정의 Kiwi 인스턴스는 재사용됩니다.

    매개변수:
        model_type: 언어 모델 타입 ('knlm', 'sbg', 'cong', 'cong-global', 'none', 'largest')
        typos: 오타 교정 모드 ('basic', 'continual', 'basic_with_continual',
            'lengthening', 'basic_with_continual_and_lengthening')
        model_path: 모델 경로 (CoNg 사용 시 필요)
        load_default_dict: 기본 사전 로드 여부 (위키백과 표제어)
        load_typo_dict: 내장 오타 사전 로드 여부
        load_multi_dict: 다어절 사전 로드 여부 (WikiData 고유명사)
        integrate_allomorph: 음운론적 이형태 통합 여부 (아/어, 았/었 등)
        typo_cost_threshold: 오타 교정 최대 비용 (기본값: 2.5)
        num_workers: 워커 수 (-1=모든 코어, 0=단일 스레드)

    반환값:
        Kiwi 인스턴스

    예시:
        >>> # 기본 설정
        >>> kiwi = get_kiwi_instance()
        >>> # CoNg 모델 사용
        >>> kiwi = get_kiwi_instance(model_type="cong", model_path="./cong-base")
        >>> # 오타 교정 활성화
        >>> kiwi = get_kiwi_instance(typos="basic_with_continual_and_lengthening")
    """
    cache_key = (
        f"{model_type}_{typos}_{model_path}_{load_default_dict}_"
        f"{load_typo_dict}_{load_multi_dict}_{integrate_allomorph}_"
        f"{typo_cost_threshold}_{num_workers}"
    )

    if cache_key not in _KIWI_INSTANCES:
        try:
            _KIWI_INSTANCES[cache_key] = Kiwi(
                model_type=model_type,
                typos=typos,
                model_path=model_path,
                load_default_dict=load_default_dict,
                load_typo_dict=load_typo_dict,
                load_multi_dict=load_multi_dict,
                integrate_allomorph=integrate_allomorph,
                typo_cost_threshold=typo_cost_threshold,
                num_workers=num_workers,
            )
        except OSError as e:
            # 모델 파일을 찾을 수 없는 경우 기본 Kiwi() 사용
            if "Cannot open" in str(e):
                print(f"⚠️  Kiwi 모델 파일을 찾을 수 없습니다: {e}")
                print("   → 기본 설정으로 Kiwi를 초기화합니다.")
                _KIWI_INSTANCES[cache_key] = Kiwi(
                    typos=typos,
                    load_default_dict=load_default_dict,
                    load_typo_dict=load_typo_dict,
                    load_multi_dict=load_multi_dict,
                    integrate_allomorph=integrate_allomorph,
                    typo_cost_threshold=typo_cost_threshold,
                    num_workers=num_workers,
                )
            else:
                raise

    return _KIWI_INSTANCES[cache_key]


def advanced_kiwi_tokenizer(
    text: str,
    kiwi: Kiwi,
    important_pos: set[str] | None = None,
    stopwords: Stopwords | None = None,
    normalize_coda: bool = False,
    z_coda: bool = True,
    compatible_jamo: bool = False,
    saisiot: bool | None = None,
    min_token_len: int = 1,
    blocklist: set[tuple[str, str]] | None = None,
) -> list[str]:
    """Kiwi의 고급 기능을 모두 활용하는 토크나이저.

    세종 품사 태그를 기반으로 의미 있는 단어만 추출하여 BM25 검색 성능을 향상시킵니다.
    기본적으로 명사, 동사, 형용사, 영어, 한자, 숫자만 포함합니다.

    매개변수:
        text: 토큰화할 텍스트
        kiwi: Kiwi 인스턴스
        important_pos: 사용할 품사 집합 (None이면 기본값 사용)
        - 기본값: get_default_important_pos() 반환값
        - 편의 함수 사용 예:
            * get_noun_only_pos(): 명사만
            * get_content_word_pos(): 명사, 동사, 형용사
            * get_all_content_pos(): 모든 내용어
        - 세종 품사 태그 참고:
            * NNG: 일반 명사, NNP: 고유 명사, NNB: 의존 명사
            * VV: 동사, VA: 형용사
            * SL: 알파벳, SH: 한자, SN: 숫자
            * 전체 태그는 SEJONG_POS_TAGS 상수 참고
        stopwords: 불용어 객체 (kiwipiepy.utils.Stopwords)
        normalize_coda: 덧붙은 받침 정규화 (ㅋㅋㅋ 처리)
            True일 경우 "안 먹었엌ㅋㅋ" -> "안 먹었어 ㅋㅋㅋ"로 정규화
        z_coda: 덧붙은 받침을 Z_CODA 태그로 분리
            True일 경우 덧붙은 받침을 별도 토큰으로 분리
        compatible_jamo: 받침을 호환용 자모로 출력
            True일 경우 받침을 자모 단위로 분해
        saisiot: 사이시옷 처리
            - True: 사이시옷을 분리 (예: "나뭇잎" -> "나무/시옷/잎")
            - False: 사이시옷을 통합
            - None: 기본값 사용
        min_token_len: 최소 토큰 길이 (기본값: 1)
        blocklist: 제외할 (형태, 품사) 쌍 집합
            예: {("것", "NNB"), ("수", "NNB")}  # 의존 명사 제외

    반환값:
        토큰 리스트 (품사 필터링 및 불용어 제거 후)

    예시:
        >>> from kiwipiepy import Kiwi
        >>> from kiwipiepy.utils import Stopwords
        >>> kiwi = Kiwi()
        >>>
        >>> # 기본 토큰화 (명사, 동사, 형용사, 영어, 한자, 숫자)
        >>> advanced_kiwi_tokenizer("한국어를 분석합니다", kiwi)
        ['한국어', '분석']
        >>>
        >>> # 명사만 추출
        >>> from day2.retrieval.kiwi_bm25_advanced import get_noun_only_pos
        >>> advanced_kiwi_tokenizer("한국어를 분석합니다", kiwi, important_pos=get_noun_only_pos())
        ['한국어']
        >>>
        >>> # 불용어 제거
        >>> stopwords = Stopwords()
        >>> advanced_kiwi_tokenizer("이것은 테스트입니다", kiwi, stopwords=stopwords)
        ['테스트']
        >>>
        >>> # 오타 정규화
        >>> advanced_kiwi_tokenizer("안 먹었엌ㅋㅋ", kiwi, normalize_coda=True)
        ['안', '먹', '었', '어', 'ㅋㅋㅋ']
        >>>
        >>> # 내용어만 (명사, 동사, 형용사)
        >>> from day2.retrieval.kiwi_bm25_advanced import get_content_word_pos
        >>> advanced_kiwi_tokenizer("빠르게 달린다", kiwi, important_pos=get_content_word_pos())
        ['빠르', '달리']
    """
    if important_pos is None:
        # 기본: 명사, 동사, 형용사, 영어, 한자, 숫자
        important_pos = get_default_important_pos()

    # Kiwi tokenize 호출
    tokens = kiwi.tokenize(
        text,
        stopwords=stopwords,
        normalize_coda=normalize_coda,
        z_coda=z_coda,
        compatible_jamo=compatible_jamo,
        saisiot=saisiot,
    )

    # 품사 필터링 및 토큰 추출
    result = []
    for token in tokens:
        # 품사 체크
        if token.tag not in important_pos:
            continue

        # Blocklist 체크
        if blocklist and (token.form, token.tag) in blocklist:
            continue

        # 최소 길이 필터
        if len(token.form) >= min_token_len:
            result.append(token.form)

    return result


class KiwiBM25Retriever(BaseRetriever):
    """Kiwi 기능을 모두 활용하는 BM25 Retriever.

    LangChain BaseRetriever를 직접 상속하며, Kiwi v0.6x의 모든 기능을 지원합니다.

    속성:
        vectorizer: BM25Okapi 인스턴스
        docs: 문서 리스트
        k: 반환할 문서 수

        # Kiwi 설정
        model_type: 언어 모델 ('knlm', 'sbg', 'cong')
        typos: 오타 교정 모드
        model_path: 모델 경로
        load_default_dict: 기본 사전 로드 여부
        num_workers: 워커 수

        # 토큰화 설정
        important_pos: 사용할 품사 집합
        stopwords: 불용어 객체
        normalize_coda: 덧붙은 받침 정규화
        z_coda: 덧붙은 받침 분리
        compatible_jamo: 호환용 자모 사용
        saisiot: 사이시옷 처리
        min_token_len: 최소 토큰 길이
        space_tolerance: 공백 허용도

        # BM25 파라미터
        bm25_k1: BM25 k1 파라미터
        bm25_b: BM25 b 파라미터

    예시:
        >>> # 기본 사용
        >>> retriever = KiwiBM25Retriever.from_documents(docs)
        >>> results = retriever.invoke("검색 쿼리")

        >>> # CoNg 모델 + 오타 교정
        >>> retriever = KiwiBM25Retriever.from_documents(docs, model_type="cong", typos="basic_with_continual")

        >>> # 명사만 + 불용어 제거
        >>> stopwords = Stopwords()
        >>> retriever = KiwiBM25Retriever.from_documents(
        ...     docs, important_pos={"NNG", "NNP"}, stopwords=stopwords
        ... )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorizer: Any = Field(default=None, repr=False)
    """BM25 vectorizer"""

    docs: list[Document] = Field(default_factory=list, repr=False)
    """문서 리스트"""

    k: int = 5
    """반환할 문서 수"""

    # Kiwi 설정
    model_type: Literal["knlm", "sbg", "cong"] = "knlm"
    """언어 모델 타입
    - 'knlm': 기본 언어 모델 (권장)
    - 'sbg': SkipBigram 모델
    - 'cong': CoNg 모델 (고정밀도, model_path 필요)
    """

    typos: (
        Literal[
            "basic",
            "continual",
            "basic_with_continual",
            "lengthening",
            "basic_with_continual_and_lengthening",
        ]
        | None
    ) = "basic_with_continual_and_lengthening"
    """오타 교정 모드
    - 'basic': 기본 오타 교정
    - 'continual': 연철 오타 교정
    - 'basic_with_continual': 기본 + 연철
    - 'lengthening': 장음화 오타 교정
    - 'basic_with_continual_and_lengthening': 모든 오타 교정 (권장)
    """

    model_path: str | None = None
    """모델 경로 (CoNg 사용 시 필요)"""

    load_default_dict: bool = True
    """기본 사전 로드 (위키백과 표제어)"""

    load_typo_dict: bool = True
    """내장 오타 사전 로드 여부"""

    load_multi_dict: bool = True
    """다어절 사전 로드 여부 (WikiData 고유명사)"""

    integrate_allomorph: bool = True
    """음운론적 이형태 통합 여부 (아/어, 았/었 등)"""

    typo_cost_threshold: float = 2.5
    """오타 교정 최대 비용"""

    num_workers: int = -1
    """워커 수 (-1=모든 코어, 0=단일 스레드)"""

    # 토큰화 설정
    important_pos: set[str] | None = None
    """사용할 품사 집합"""

    stopwords: Stopwords | None = None
    """불용어 객체 (Stopwords)"""

    normalize_coda: bool = False
    """덧붙은 받침 정규화 (ㅋㅋㅋ 처리)"""

    z_coda: bool = True
    """덧붙은 받침을 Z_CODA로 분리"""

    compatible_jamo: bool = False
    """받침을 호환용 자모로 출력"""

    saisiot: bool | None = None
    """사이시옷 처리 (True: 분리, False: 통합)"""

    min_token_len: int = 1
    """최소 토큰 길이"""

    space_tolerance: int = 0
    """공백 허용도 (형태소 내 공백 허용 개수)"""

    blocklist: set[tuple[str, str]] | None = None
    """제외할 (형태, 품사) 쌍 집합"""

    # NgramExtractor 기반 다어절 보강 설정
    enable_ngram_enrichment: bool = False
    """NgramExtractor를 사용한 다어절 보강 활성화 여부"""

    ngram_min_cnt: int = 10
    """Ngram 추출 최소 출현 빈도"""

    ngram_max_length: int = 5
    """Ngram 최대 길이"""

    ngram_min_score: float = 1e-3
    """Ngram 최소 점수"""

    ngram_auto_add: bool = True
    """추출된 Ngram을 자동으로 사용자 사전에 추가할지 여부"""

    # BM25 파라미터
    bm25_k1: float = 1.5
    """BM25 k1 파라미터"""

    bm25_b: float = 0.75
    """BM25 b 파라미터"""

    # 내부 상태
    _kiwi: Kiwi | None = None
    """Kiwi 인스턴스 (내부용)"""

    def _get_kiwi(self) -> Kiwi:
        """Kiwi 인스턴스를 가져온다 (lazy initialization)."""
        if self._kiwi is None:
            self._kiwi = get_kiwi_instance(
                model_type=self.model_type,
                typos=self.typos,
                model_path=self.model_path,
                load_default_dict=self.load_default_dict,
                load_typo_dict=self.load_typo_dict,
                load_multi_dict=self.load_multi_dict,
                integrate_allomorph=self.integrate_allomorph,
                typo_cost_threshold=self.typo_cost_threshold,
                num_workers=self.num_workers,
            )

            # space_tolerance 설정
            if self.space_tolerance > 0:
                self._kiwi.space_tolerance = self.space_tolerance

        return self._kiwi

    def add_user_word(
        self,
        word: str,
        tag: str = "NNP",
        score: float = 0.0,
        orig_word: str | None = None,
    ) -> bool:
        """사용자 사전에 단어를 추가한다.

        매개변수:
            word: 추가할 단어
            tag: 품사 태그
            score: 점수
            orig_word: 원본 단어 (이형태인 경우)

        반환값:
            추가 성공 여부

        예시:
            >>> retriever.add_user_word("김갑갑", "NNP")
            True
            >>> retriever.add_user_word("팅기", "VV")  # 동사 활용형 자동 등재
            True
        """
        kiwi = self._get_kiwi()
        return kiwi.add_user_word(word, tag, score, orig_word)

    def add_pre_analyzed_word(
        self,
        form: str,
        analyzed: list[str | tuple],
        score: float = 0.0,
    ) -> bool:
        """기분석 형태를 추가한다.

        매개변수:
            form: 기분석 형태
            analyzed: 형태소 분석 결과
            score: 점수

        반환값:
            추가 성공 여부

        예시:
            >>> retriever.add_pre_analyzed_word(
            ...     "사겼다", [("사귀", "VV", 0, 2), ("었", "EP", 1, 2), ("다", "EF", 2, 3)], -3.0
            ... )
            True
        """
        kiwi = self._get_kiwi()
        return kiwi.add_pre_analyzed_word(form, analyzed, score)

    def add_re_rule(
        self,
        tag: str,
        pattern: str,
        repl: str,
        score: float = -3.0,
    ) -> list[str]:
        """정규표현식 규칙으로 이형태를 일괄 추가한다.

        매개변수:
            tag: 품사 태그
            pattern: 정규표현식 패턴
            repl: 치환 문자열
            score: 점수

        반환값:
            추가된 형태소 리스트

        예시:
            >>> # '요' -> '영' 종결어미 변형 일괄 추가
            >>> retriever.add_re_rule("EF", r"요$", r"영", -3.0)
            ['어영', '에영', '지영', ...]
        """
        kiwi = self._get_kiwi()
        return kiwi.add_re_rule(tag, pattern, repl, score)

    def load_user_dictionary(self, user_dict_path: str) -> int:
        """사용자 사전 파일을 로드한다.

        매개변수:
            user_dict_path: 사전 파일 경로

        반환값:
            추가된 형태소 개수
        """
        kiwi = self._get_kiwi()
        return kiwi.load_user_dictionary(user_dict_path)

    def _enrich_with_ngrams(self, texts: list[str]) -> None:
        """NgramExtractor를 사용하여 다어절 단어를 추출하고 사전에 추가한다.

        매개변수:
            texts: 분석할 텍스트 리스트
        """
        if not self.enable_ngram_enrichment:
            return

        kiwi = self._get_kiwi()
        extractor = NgramExtractor(kiwi, gather_lm_score=True)

        # 텍스트 추가
        extractor.add(texts)

        # Ngram 추출
        candidates = extractor.extract(
            max_candidates=-1,
            min_cnt=self.ngram_min_cnt,
            max_length=self.ngram_max_length,
            min_score=self.ngram_min_score,
            num_workers=self.num_workers if self.num_workers > 0 else 1,
        )

        # 자동 추가 옵션이 활성화된 경우 사용자 사전에 추가
        if self.ngram_auto_add:
            added_count = 0
            for candidate in candidates:
                # candidate.text는 문자열 형태의 Ngram
                # candidate.tokens는 (형태, 품사) 튜플 리스트
                if candidate.text and len(candidate.text.strip()) > 0:
                    # 공백이 포함된 다어절 단어는 NNP(고유명사)로 추가
                    # 또는 길이가 2 이상인 단어도 추가
                    text = candidate.text.strip()
                    if " " in text or len(text) > 2:
                        if kiwi.add_user_word(text, "NNP", score=0.0):
                            added_count += 1

            if added_count > 0:
                print(f"NgramExtractor로 {added_count}개 다어절 단어 추가됨")

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        metadatas: list[dict] | None = None,
        *,
        auto_save: bool = False,
        save_path: str | Path | None = None,
        save_user_dict: bool = True,
        **kwargs: Any,
    ) -> "KiwiBM25Retriever":
        """텍스트 리스트로부터 Retriever 생성.

        매개변수:
            texts: 텍스트 리스트
            metadatas: 메타데이터 리스트
            auto_save: 생성 직후 인덱스를 자동 저장
            save_path: 자동 저장 경로 (미지정 시 'models/kiwi_bm25')
            save_user_dict: 사용자 사전 템플릿도 함께 저장
            **kwargs: 추가 파라미터
                - enable_ngram_enrichment: NgramExtractor 기반 다어절 보강 활성화
                - ngram_min_cnt: Ngram 최소 출현 빈도 (기본값: 10)
                - ngram_max_length: Ngram 최대 길이 (기본값: 5)
                - ngram_min_score: Ngram 최소 점수 (기본값: 1e-3)
                - ngram_auto_add: 추출된 Ngram 자동 추가 여부 (기본값: True)

        반환값:
            KiwiBM25Retriever 인스턴스

        예시:
            >>> # 기본 사용
            >>> retriever = KiwiBM25Retriever.from_texts(texts)
            >>>
            >>> # Ngram 보강 활성화
            >>> retriever = KiwiBM25Retriever.from_texts(texts, enable_ngram_enrichment=True, ngram_min_cnt=5)
        """
        # Retriever 인스턴스 생성 (Kiwi는 lazy initialization)
        instance = cls(**kwargs)
        kiwi = instance._get_kiwi()

        # Ngram 보강 수행 (사전에 단어 추가)
        instance._enrich_with_ngrams(texts)

        # 커스텀 토크나이저 생성
        def custom_tokenizer(text: str) -> list[str]:
            return advanced_kiwi_tokenizer(
                text,
                kiwi=kiwi,
                important_pos=instance.important_pos,
                stopwords=instance.stopwords,
                normalize_coda=instance.normalize_coda,
                z_coda=instance.z_coda,
                compatible_jamo=instance.compatible_jamo,
                saisiot=instance.saisiot,
                min_token_len=instance.min_token_len,
                blocklist=instance.blocklist,
            )

        # 텍스트 토큰화
        texts_processed = [custom_tokenizer(t) for t in texts]

        # BM25 인덱스 생성
        instance.vectorizer = BM25Okapi(
            texts_processed,
            k1=instance.bm25_k1,
            b=instance.bm25_b,
        )

        # 문서 생성
        metadatas = metadatas or [{} for _ in texts]
        instance.docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas, strict=True)]

        # 자동 저장
        if auto_save:
            target_path = Path(save_path) if save_path is not None else Path("models/kiwi_bm25")
            instance.save(target_path, save_user_dict=save_user_dict)

        return instance

    @classmethod
    def from_documents(
        cls,
        documents: Any,
        *,
        auto_save: bool = False,
        save_path: str | Path | None = None,
        save_user_dict: bool = True,
        load_user_dict: bool = True,
        **kwargs: Any,
    ) -> "KiwiBM25Retriever":
        """입력으로부터 Retriever 생성 (Document 리스트, 텍스트 리스트, 또는 저장 경로).

        매개변수:
            documents: 아래 중 하나
                - list[Document]: 문서 리스트
                - list[str]: 텍스트 리스트
                - str | Path: 저장된 인덱스 디렉토리 경로 (로드)
            auto_save: 인덱스 생성 후 자동 저장
            save_path: 자동 저장 경로 (미지정 시 'models/kiwi_bm25')
            save_user_dict: 자동 저장 시 사용자 사전 템플릿도 저장
            load_user_dict: 로드시 사용자 사전 로드 여부
            **kwargs: from_texts의 나머지 파라미터들과 동일

        반환값:
            KiwiBM25Retriever 인스턴스

        예시:
            >>> # 1) Document 리스트로 생성
            >>> retriever = KiwiBM25Retriever.from_documents(docs, auto_save=True, save_path="models/my_ret")
            >>>
            >>> # 2) 텍스트 리스트로 생성
            >>> retriever = KiwiBM25Retriever.from_documents(["문장1", "문장2"], auto_save=True)
            >>>
            >>> # 3) 저장된 인덱스 로드
            >>> retriever = KiwiBM25Retriever.from_documents("models/my_ret")
        """
        # 3) 경로 입력 시 로드
        if isinstance(documents, (str, Path)):
            return cls.load(documents, load_user_dict=load_user_dict)

        # 2) 텍스트 리스트 처리
        if isinstance(documents, list) and (len(documents) == 0 or isinstance(documents[0], str)):
            texts = documents  # type: ignore[assignment]
            return cls.from_texts(
                texts=texts,  # type: ignore[arg-type]
                metadatas=None,
                auto_save=auto_save,
                save_path=save_path,
                save_user_dict=save_user_dict,
                **kwargs,
            )

        # 1) Document 리스트 처리
        if isinstance(documents, list) and documents and isinstance(documents[0], Document):
            texts = [d.page_content for d in documents]
            metadatas = [d.metadata for d in documents]
            return cls.from_texts(
                texts=texts,
                metadatas=metadatas,
                auto_save=auto_save,
                save_path=save_path,
                save_user_dict=save_user_dict,
                **kwargs,
            )

        raise TypeError("documents must be list[Document], list[str], or a path (str|Path) to a saved index")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """관련 문서 검색.
        점수를 메타데이터에 포함하여 반환합니다.
        """
        kiwi = self._get_kiwi()

        # 쿼리 토큰화
        processed_query = advanced_kiwi_tokenizer(
            query,
            kiwi=kiwi,
            important_pos=self.important_pos,
            stopwords=self.stopwords,
            normalize_coda=self.normalize_coda,
            z_coda=self.z_coda,
            compatible_jamo=self.compatible_jamo,
            saisiot=self.saisiot,
            min_token_len=self.min_token_len,
            blocklist=self.blocklist,
        )

        # BM25 점수 계산
        scores = self.vectorizer.get_scores(processed_query)
        
        # Softmax 정규화
        normalized_scores = self._softmax(scores)
        
        # 점수 내림차순 정렬
        score_indices = self._argsort(normalized_scores, reverse=True)[:self.k]
        
        # 문서에 점수 추가
        docs_with_scores = []
        for idx in score_indices:
            doc = self.docs[idx]
            metadata = doc.metadata.copy()
            metadata["score"] = float(normalized_scores[idx])
            docs_with_scores.append(Document(page_content=doc.page_content, metadata=metadata))
        
        return docs_with_scores

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """관련 문서를 비동기적으로 검색."""

        sync_manager = run_manager.get_sync() if run_manager is not None else None
        return await asyncio.to_thread(
            self._get_relevant_documents,
            query,
            run_manager=sync_manager,
        )

    def search_with_score(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[Document]:
        """점수와 함께 검색 (메타데이터에 score 추가).

        매개변수:
            query: 검색 쿼리
            top_k: 반환할 문서 수 (None이면 self.k 사용)

        반환값:
            점수가 메타데이터에 포함된 문서 리스트
        """
        if top_k is None:
            top_k = self.k

        kiwi = self._get_kiwi()

        # 쿼리 토큰화
        processed_query = advanced_kiwi_tokenizer(
            query,
            kiwi=kiwi,
            important_pos=self.important_pos,
            stopwords=self.stopwords,
            normalize_coda=self.normalize_coda,
            z_coda=self.z_coda,
            compatible_jamo=self.compatible_jamo,
            saisiot=self.saisiot,
            min_token_len=self.min_token_len,
            blocklist=self.blocklist,
        )

        # BM25 점수 계산
        scores = self.vectorizer.get_scores(processed_query)

        # Softmax 정규화
        normalized_scores = self._softmax(scores)

        # 점수 내림차순 정렬
        score_indices = self._argsort(normalized_scores, reverse=True)[:top_k]

        # 문서에 점수 추가
        docs_with_scores = []
        for idx in score_indices:
            doc = self.docs[idx]
            metadata = doc.metadata.copy()
            metadata["score"] = float(normalized_scores[idx])
            docs_with_scores.append(Document(page_content=doc.page_content, metadata=metadata))

        return docs_with_scores

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax 정규화."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @staticmethod
    def _argsort(seq: list, reverse: bool = False) -> list:
        """시퀀스 정렬 인덱스 반환."""
        return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)

    def save(
        self,
        path: str | Path,
        save_user_dict: bool = True,
    ) -> None:
        """Retriever 인덱스를 저장한다.

        BM25 인덱스와 문서, 토큰화 결과를 저장하여
        다음에 빠르게 로드할 수 있습니다.

        매개변수:
            path: 저장할 디렉토리 경로
            save_user_dict: 사용자 사전도 함께 저장 (True 권장)

        예시:
            >>> retriever.save("models/my_retriever")
            >>> # 다음 실행 시 빠르게 로드
            >>> retriever = KiwiBM25Retriever.load("models/my_retriever")

        참고:
            Kiwi 모델 자체는 C++ 바인딩이라 저장되지 않습니다.
            사용자 사전은 텍스트 파일로 별도 저장됩니다.
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 1. BM25 인덱스 + 문서 저장
        index_file = save_path / "bm25_index.pkl"
        with open(index_file, "wb") as f:
            pickle.dump(
                {
                    "docs": self.docs,
                    "vectorizer": self.vectorizer,
                    # 설정 저장
                    "k": self.k,
                    "model_type": self.model_type,
                    "typos": self.typos,
                    "load_default_dict": self.load_default_dict,
                    "load_typo_dict": self.load_typo_dict,
                    "load_multi_dict": self.load_multi_dict,
                    "integrate_allomorph": self.integrate_allomorph,
                    "typo_cost_threshold": self.typo_cost_threshold,
                    "important_pos": self.important_pos,
                    "normalize_coda": self.normalize_coda,
                    "z_coda": self.z_coda,
                    "compatible_jamo": self.compatible_jamo,
                    "saisiot": self.saisiot,
                    "min_token_len": self.min_token_len,
                    "space_tolerance": self.space_tolerance,
                    "enable_ngram_enrichment": self.enable_ngram_enrichment,
                    "ngram_min_cnt": self.ngram_min_cnt,
                    "ngram_max_length": self.ngram_max_length,
                    "ngram_min_score": self.ngram_min_score,
                    "ngram_auto_add": self.ngram_auto_add,
                    "bm25_k1": self.bm25_k1,
                    "bm25_b": self.bm25_b,
                },
                f,
            )

        # 2. 사용자 사전 저장 (선택)
        if save_user_dict:
            # Kiwi는 사용자 사전 직접 저장 기능이 없으므로
            # 메타데이터로 저장 안내만 출력
            dict_file = save_path / "user_dict.txt"
            if not dict_file.exists():
                with open(dict_file, "w", encoding="utf-8") as f:
                    f.write("# 사용자 사전 파일\n")
                    f.write("# 형태\\t품사\\t점수 형식으로 추가하세요\n")
                    f.write("# 예: 김갑갑\\tNNP\\t0.0\n")

        print(f"Retriever 저장 완료: {save_path}")
        print(f"  - BM25 인덱스: {index_file}")
        if save_user_dict:
            print(f"  - 사용자 사전: {dict_file} (수동 편집 필요)")

    @classmethod
    def load(
        cls,
        path: str | Path,
        load_user_dict: bool = True,
    ) -> "KiwiBM25Retriever":
        """저장된 Retriever 인덱스를 로드한다.

        매개변수:
            path: 로드할 디렉토리 경로
            load_user_dict: 사용자 사전도 로드

        반환값:
            로드된 KiwiBM25Retriever 인스턴스

        예시:
            >>> retriever = KiwiBM25Retriever.load("models/my_retriever")
            >>> results = retriever.invoke("검색 쿼리")

        참고:
            저장된 인덱스를 로드하면 문서 재분석 없이 바로 검색 가능합니다.
            사용자 사전은 user_dict.txt를 수동으로 편집 후 로드됩니다.
        """
        load_path = Path(path)
        index_file = load_path / "bm25_index.pkl"

        if not index_file.exists():
            raise FileNotFoundError(f"인덱스 파일을 찾을 수 없습니다: {index_file}")

        # 1. BM25 인덱스 로드
        with open(index_file, "rb") as f:
            data = pickle.load(f)

        # 2. Retriever 인스턴스 생성
        instance = cls(
            k=data.get("k", 4),
            model_type=data.get("model_type", "knlm"),
            typos=data.get("typos"),
            load_default_dict=data.get("load_default_dict", True),
            load_typo_dict=data.get("load_typo_dict", True),
            load_multi_dict=data.get("load_multi_dict", True),
            integrate_allomorph=data.get("integrate_allomorph", True),
            typo_cost_threshold=data.get("typo_cost_threshold", 2.5),
            important_pos=data.get("important_pos"),
            normalize_coda=data.get("normalize_coda", False),
            z_coda=data.get("z_coda", True),
            compatible_jamo=data.get("compatible_jamo", False),
            saisiot=data.get("saisiot"),
            min_token_len=data.get("min_token_len", 1),
            space_tolerance=data.get("space_tolerance", 0),
            enable_ngram_enrichment=data.get("enable_ngram_enrichment", False),
            ngram_min_cnt=data.get("ngram_min_cnt", 10),
            ngram_max_length=data.get("ngram_max_length", 5),
            ngram_min_score=data.get("ngram_min_score", 1e-3),
            ngram_auto_add=data.get("ngram_auto_add", True),
            bm25_k1=data.get("bm25_k1", 1.5),
            bm25_b=data.get("bm25_b", 0.75),
        )

        # 3. 데이터 복원
        instance.docs = data["docs"]
        instance.vectorizer = data["vectorizer"]

        # 4. 사용자 사전 로드 (선택)
        if load_user_dict:
            dict_file = load_path / "user_dict.txt"
            if dict_file.exists():
                try:
                    kiwi = instance._get_kiwi()
                    num_added = kiwi.load_user_dictionary(str(dict_file))
                    print(f"사용자 사전 로드: {num_added}개 단어")
                except Exception as e:
                    print(f"사용자 사전 로드 실패: {e}")

        print(f"Retriever 로드 완료: {load_path}")
        print(f" - 문서 수: {len(instance.docs)}")
        print(f" - 모델: {instance.model_type}")

        return instance


# 편의 함수
def create_kiwi_bm25_retriever(
    documents: list[Document],
    model_type: Literal["knlm", "sbg", "cong"] = "knlm",
    **kwargs: Any,
) -> KiwiBM25Retriever:
    """KiwiBM25Retriever 생성 헬퍼 함수.

    예시:
        >>> retriever = create_kiwi_bm25_retriever(docs, model_type="sbg")
    """
    return KiwiBM25Retriever.from_documents(
        documents,
        model_type=model_type,
        **kwargs,
    )
