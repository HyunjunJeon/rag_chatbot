"""
BM25 인덱스 재구축 스크립트

이 스크립트는 기존의 merged JSON 데이터를 LangChain Document 형식으로 변환하고,
KiwiBM25Retriever를 사용하여 BM25 인덱스를 재구축합니다.

사용법:
    python document_processing/rebuild_bm25_for_chatbot.py
"""

import json
import sys
from pathlib import Path
from typing import Any

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

# LangChain Document import
from langchain_core.documents import Document
from tqdm import tqdm

# KiwiBM25Retriever를 직접 import (설정 파일 로드 우회)
import importlib.util

# kiwi_bm25_retriever 모듈 직접 로드
spec = importlib.util.spec_from_file_location(
    "kiwi_bm25_retriever",
    PROJECT_ROOT / "app" / "naver_connect_chatbot" / "rag" / "retriever" / "kiwi_bm25_retriever.py"
)
kiwi_module = importlib.util.module_from_spec(spec)
sys.modules['kiwi_bm25_retriever'] = kiwi_module
spec.loader.exec_module(kiwi_module)

KiwiBM25Retriever = kiwi_module.KiwiBM25Retriever
get_default_important_pos = kiwi_module.get_default_important_pos

__all__ = ["rebuild_bm25_index"]


def load_merged_json(json_path: Path) -> dict[str, Any]:
    """
    merged JSON 파일을 로드합니다.

    매개변수:
        json_path: JSON 파일 경로

    반환값:
        JSON 데이터 딕셔너리
    """
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def convert_to_documents(merged_dir: Path) -> list[Document]:
    """
    merged JSON 디렉토리의 모든 데이터를 LangChain Document로 변환합니다.

    매개변수:
        merged_dir: merged JSON 디렉토리 경로

    반환값:
        Document 리스트
    """
    print(f"\n📂 디렉토리: {merged_dir}")

    # JSON 파일 찾기
    json_files = sorted(merged_dir.glob("*_merged.json"))
    print(f"   발견된 파일: {len(json_files)}개")

    documents: list[Document] = []

    for json_file in tqdm(json_files, desc="파일 처리"):
        # _summary.json은 건너뛰기
        if json_file.name == "_summary.json":
            continue

        try:
            data = load_merged_json(json_file)
            course = data["course"]

            for qa in data["qa_pairs"]:
                question = qa["question"]
                answers = qa["answers"]

                # 각 답변마다 Document 생성
                for idx, answer in enumerate(answers):
                    # Document ID 생성
                    doc_id = (
                        f"{qa['generation']}_{course}_{qa['date']}_"
                        f"{question['timestamp']}_a{idx}"
                    )

                    # page_content: 질문과 답변 결합
                    page_content = f"질문: {question['text']}\n답변: {answer['text']}"

                    # metadata: 검색 필터링 및 추적용
                    metadata = {
                        "doc_id": doc_id,
                        "course": course,
                        "generation": qa["generation"],
                        "date": qa["date"],
                        "question_text": question["text"],
                        "answer_text": answer["text"],
                        "question_user": question.get("user_name") or question["user"],
                        "answer_user": answer.get("user_name") or answer["user"],
                        "is_bot": answer["is_bot"],
                        "question_timestamp": question["timestamp"],
                        "answer_timestamp": answer["timestamp"],
                    }

                    # Document 생성
                    doc = Document(page_content=page_content, metadata=metadata)
                    documents.append(doc)

        except Exception as e:
            print(f"\n   ✗ {json_file.name} 처리 실패: {e}")
            continue

    print(f"\n   총 문서 수: {len(documents):,}개")

    return documents


def rebuild_bm25_index(
    merged_dir: Path | str,
    output_dir: Path | str,
    k: int = 10,
    typos: str = "basic_with_continual_and_lengthening",
) -> KiwiBM25Retriever:
    """
    BM25 인덱스를 재구축합니다.

    매개변수:
        merged_dir: merged JSON 디렉토리 경로
        output_dir: 출력 디렉토리 경로
        k: 기본 검색 결과 수
        typos: Kiwi 오타 교정 모드

    반환값:
        KiwiBM25Retriever 인스턴스
    """
    merged_dir = Path(merged_dir)
    output_dir = Path(output_dir)

    print("=" * 80)
    print("🚀 BM25 인덱스 재구축 시작")
    print("=" * 80)

    # 1. Document 변환
    print("\n[1] JSON 데이터를 Document로 변환")
    documents = convert_to_documents(merged_dir)

    if not documents:
        raise ValueError("변환된 문서가 없습니다.")

    # 2. KiwiBM25Retriever 생성
    print("\n[2] KiwiBM25Retriever 생성")
    print(f"   K: {k}")
    print(f"   오타 교정: {typos}")
    print("   품사 필터: 명사, 동사, 형용사, 영어, 한자, 숫자")

    retriever = KiwiBM25Retriever.from_documents(
        documents=documents,
        k=k,
        model_type="knlm",  # KNLM 기본 모델 사용
        typos=None,  # 오타 교정 비활성화
        important_pos=get_default_important_pos(),
        load_default_dict=True,  # 기본 사전 활성화
        load_typo_dict=False,  # 오타 사전 비활성화
        load_multi_dict=False,  # 다어절 사전 비활성화
        num_workers=0,  # 단일 스레드 실행
        auto_save=True,
        save_path=output_dir,
        save_user_dict=True,
    )

    print("\n   ✅ BM25 인덱스 생성 완료")
    print(f"      저장 위치: {output_dir}")

    # 3. 통계 출력
    print("\n[3] 인덱스 통계")
    print(f"   총 문서 수: {len(retriever.docs):,}개")

    # 과정별 통계
    course_counts: dict[str, int] = {}
    for doc in retriever.docs:
        course = doc.metadata["course"]
        course_counts[course] = course_counts.get(course, 0) + 1

    print(f"   과정 수: {len(course_counts)}개")
    for course, count in sorted(course_counts.items()):
        print(f"      - {course}: {count:,}개")

    # 4. 간단한 검색 테스트
    print("\n[4] 검색 테스트")
    test_queries = [
        "GPU 메모리 부족",
        "데이터 증강",
        "optimizer",
    ]

    for query in test_queries:
        print(f"\n   쿼리: '{query}'")
        results = retriever.invoke(query)
        
        if results:
            top_result = results[0]
            print(f"      - 상위 결과: {top_result.metadata['course']}")
            print(f"      - 질문: {top_result.metadata['question_text'][:50]}...")
            print(f"      - 답변: {top_result.metadata['answer_text'][:50]}...")
            if "score" in top_result.metadata:
                print(f"      - 점수: {top_result.metadata['score']:.4f}")

    print("\n" + "=" * 80)
    print("✅ BM25 인덱스 재구축 완료!")
    print("=" * 80)

    return retriever


def main() -> None:
    """메인 함수"""
    # 경로 설정
    merged_dir = PROJECT_ROOT / "document_chunks" / "slack_qa_merged"
    output_dir = PROJECT_ROOT / "sparse_index" / "kiwi_bm25_slack_qa"

    # 인덱스 재구축
    try:
        retriever = rebuild_bm25_index(
            merged_dir=merged_dir,
            output_dir=output_dir,
            k=10,
        )

        print("\n💡 다음 단계:")
        print("   1. 통합 테스트 실행: python tests/test_integration_retriever.py")
        print("   2. Hybrid Retriever 구성 및 테스트")
        print(f"\n📁 저장된 파일:")
        print(f"   - {output_dir / 'bm25_index.pkl'}")
        print(f"   - {output_dir / 'user_dict.txt'}")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

