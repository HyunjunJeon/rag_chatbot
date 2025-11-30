#!/usr/bin/env python3
"""
BM25 통합 인덱스 재생성 스크립트

Qdrant VectorDB에서 전체 문서를 추출하여 Kiwi 기반 통합 BM25 인덱스를 생성합니다.

사용법:
    python document_processing/rebuild_unified_bm25.py
"""

import os
import sys
import time
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from qdrant_client import QdrantClient  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from naver_connect_chatbot.rag.retriever.kiwi_bm25_retriever import (  # noqa: E402
    KiwiBM25Retriever,
    get_default_important_pos,
)


def main():
    print("=" * 80)
    print("BM25 통합 인덱스 재생성")
    print("=" * 80)

    # Step 1: 기존 BM25 삭제
    old_index = PROJECT_ROOT / "sparse_index" / "bm25_slack_qa.pkl"
    if old_index.exists():
        os.remove(old_index)
        print(f"[1] 기존 인덱스 삭제: {old_index}")
    else:
        print("[1] 기존 인덱스 없음 (스킵)")

    # Step 2: Qdrant에서 문서 추출
    print("\n[2] Qdrant에서 문서 추출...")
    client = QdrantClient(url='http://localhost:6333')

    all_points = []
    offset = None
    while True:
        result = client.scroll(
            collection_name='naver_connect_docs',
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(result[0])
        offset = result[1]
        if offset is None:
            break
        print(f"    진행: {len(all_points)}개...", end="\r")

    print(f"    추출 완료: {len(all_points)}개 문서")

    # Step 3: Document 변환
    print("\n[3] LangChain Document 변환...")
    documents = []
    skipped = 0
    doc_type_counts = {}

    for point in all_points:
        payload = point.payload
        content = payload.get('content', '')

        if not content or not content.strip():
            skipped += 1
            continue

        metadata = {k: v for k, v in payload.items() if k != 'content'}
        metadata['qdrant_point_id'] = str(point.id)

        doc_type = metadata.get('doc_type', 'unknown')
        doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1

        documents.append(Document(page_content=content, metadata=metadata))

    print(f"    변환 완료: {len(documents)}개 (스킵: {skipped}개)")
    print("    doc_type별:")
    for dtype, count in sorted(doc_type_counts.items()):
        print(f"      - {dtype}: {count}개")

    # Step 4: BM25 인덱스 생성
    print("\n[4] KiwiBM25Retriever 생성...")
    start = time.time()

    retriever = KiwiBM25Retriever.from_documents(
        documents=documents,
        k=10,
        model_type="knlm",
        typos=None,
        important_pos=get_default_important_pos(),
        load_default_dict=True,
        load_typo_dict=False,
        load_multi_dict=False,
        num_workers=0,
        auto_save=True,
        save_path=str(PROJECT_ROOT / "sparse_index" / "unified_bm25"),
        save_user_dict=True,
    )

    build_time = time.time() - start
    print(f"    생성 완료: {build_time:.1f}초")

    # Step 5: 검증
    print("\n[5] 검증...")

    # 파일 크기
    index_path = PROJECT_ROOT / "sparse_index" / "unified_bm25" / "bm25_index.pkl"
    size_mb = os.path.getsize(index_path) / (1024 * 1024)
    print(f"    인덱스 크기: {size_mb:.2f} MB")

    # 로드 시간
    load_times = []
    for i in range(3):
        start = time.time()
        _ = KiwiBM25Retriever.load(str(PROJECT_ROOT / "sparse_index" / "unified_bm25"))
        load_times.append(time.time() - start)
    avg_load = sum(load_times) / len(load_times)
    print(f"    평균 로드 시간: {avg_load:.2f}초")

    # 검색 테스트
    test_queries = ["GPU 메모리 부족", "데이터 증강", "transformer"]
    for query in test_queries:
        start = time.time()
        results = retriever.invoke(query)
        elapsed = (time.time() - start) * 1000
        print(f"    '{query}': {elapsed:.1f}ms, {len(results)} results")

    print("\n" + "=" * 80)
    print("완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
