#!/usr/bin/env python3
"""
Qdrant Payload 인덱스 생성 스크립트

VectorDB에 저장된 데이터의 필터링 성능을 최적화하기 위해
Payload 인덱스를 생성합니다.

현재 컬렉션에 인덱스만 추가하므로 데이터 재적재가 불필요합니다.

사용법:
    python scripts/create_payload_indexes.py
    python scripts/create_payload_indexes.py --url http://qdrant:6333
    python scripts/create_payload_indexes.py --collection my_collection

참고:
    - Qdrant 공식 문서: https://qdrant.tech/documentation/concepts/indexing/
    - Tenant Indexing: https://qdrant.tech/documentation/guides/multiple-partitions/
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    KeywordIndexParams,
    PayloadSchemaType,
)


def create_indexes(
    qdrant_url: str = "http://localhost:6333",
    collection_name: str = "naver_connect_docs",
    dry_run: bool = False,
) -> None:
    """
    Qdrant 컬렉션에 Payload 인덱스를 생성합니다.

    Args:
        qdrant_url: Qdrant 서버 URL
        collection_name: 컬렉션 이름
        dry_run: True면 실제 생성 없이 계획만 출력
    """
    print("=" * 60)
    print("🔧 Qdrant Payload 인덱스 생성")
    print("=" * 60)
    print(f"   URL: {qdrant_url}")
    print(f"   Collection: {collection_name}")
    print(f"   Dry Run: {dry_run}")
    print()

    client = QdrantClient(url=qdrant_url)

    # 컬렉션 존재 확인
    try:
        info = client.get_collection(collection_name)
        print("📊 현재 컬렉션 상태:")
        print(f"   문서 수: {info.points_count:,}")
        print(f"   인덱스된 벡터: {info.indexed_vectors_count:,}")
        print(f"   현재 Payload 스키마: {info.payload_schema or '(없음)'}")
        print()
    except UnexpectedResponse as e:
        print(f"❌ 컬렉션을 찾을 수 없습니다: {collection_name}")
        print(f"   오류: {e}")
        sys.exit(1)

    # 생성할 인덱스 목록
    # Qdrant 공식 문서 권장사항:
    # "You should always create payload indexes for all fields used in filters"
    #
    # qdrant-client 1.16.0에서는 field_schema에 직접 IndexParams를 전달
    indexes_to_create = [
        # 1. course - Tenant 인덱싱 (가장 중요)
        # 과정별 조회가 대부분이므로 is_tenant=True로 데이터 co-location 최적화
        {
            "field_name": "course",
            "field_schema": KeywordIndexParams(
                type="keyword",
                is_tenant=True,  # 과정별 데이터 co-location
                on_disk=False,  # 메모리에 유지 (빈번한 필터링)
            ),
            "description": "과정별 필터링 (Tenant Index)",
            "is_tenant": True,
        },
        # 2. doc_type - 문서 타입 필터링 (필수)
        {
            "field_name": "doc_type",
            "field_schema": PayloadSchemaType.KEYWORD,
            "description": "문서 타입 필터링",
        },
        # 3. 추가 키워드 인덱스
        {
            "field_name": "difficulty",
            "field_schema": PayloadSchemaType.KEYWORD,
            "description": "난이도 필터링 (notebook)",
        },
        {
            "field_name": "instructor",
            "field_schema": PayloadSchemaType.KEYWORD,
            "description": "강사별 필터링 (pdf)",
        },
        {
            "field_name": "file_type",
            "field_schema": PayloadSchemaType.KEYWORD,
            "description": "파일 타입 필터링 (notebook: 정답/문제)",
        },
        {
            "field_name": "topic",
            "field_schema": PayloadSchemaType.KEYWORD,
            "description": "주제 필터링 (notebook)",
        },
    ]

    print(f"📋 생성할 인덱스 ({len(indexes_to_create)}개):")
    for idx in indexes_to_create:
        tenant_flag = " [TENANT]" if idx.get("is_tenant") else ""
        print(f"   • {idx['field_name']}: {idx['description']}{tenant_flag}")
    print()

    if dry_run:
        print("ℹ️ Dry run 모드 - 실제 인덱스를 생성하지 않습니다.")
        return

    # 인덱스 생성
    print("🔄 인덱스 생성 중...")
    success_count = 0
    skip_count = 0
    error_count = 0

    for idx in indexes_to_create:
        field_name = idx["field_name"]
        try:
            # qdrant-client 1.16.0 API:
            # field_schema에 PayloadSchemaType 또는 IndexParams를 직접 전달
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=idx["field_schema"],
            )
            print(f"   ✅ {field_name}")
            success_count += 1

        except UnexpectedResponse as e:
            error_msg = str(e)
            if "already exists" in error_msg.lower():
                print(f"   ⏭️ {field_name} (이미 존재)")
                skip_count += 1
            else:
                print(f"   ❌ {field_name}: {error_msg[:100]}")
                error_count += 1

    print()
    print("=" * 60)
    print("📊 결과 요약")
    print("=" * 60)
    print(f"   성공: {success_count}")
    print(f"   건너뜀: {skip_count}")
    print(f"   실패: {error_count}")
    print()

    # 최종 상태 확인
    info = client.get_collection(collection_name)
    print("📋 최종 Payload 스키마:")
    if info.payload_schema:
        for field_name, field_info in info.payload_schema.items():
            print(f"   • {field_name}: {field_info}")
    else:
        print("   (없음)")

    print()
    print("✅ 완료!")
    print()
    print("다음 단계:")
    print("   1. 검색 쿼리에 필터 적용 테스트")
    print("   2. 성능 모니터링 (쿼리 응답 시간)")
    print("   3. 필요시 추가 인덱스 생성")


def main() -> None:
    """메인 함수."""
    parser = argparse.ArgumentParser(
        description="Qdrant Payload 인덱스 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    # 기본 실행
    python scripts/create_payload_indexes.py

    # 다른 Qdrant 서버
    python scripts/create_payload_indexes.py --url http://qdrant:6333

    # 계획만 확인 (dry run)
    python scripts/create_payload_indexes.py --dry-run
        """,
    )
    parser.add_argument(
        "--url",
        default="http://localhost:6333",
        help="Qdrant 서버 URL (기본: http://localhost:6333)",
    )
    parser.add_argument(
        "--collection",
        default="naver_connect_docs",
        help="컬렉션 이름 (기본: naver_connect_docs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 인덱스 생성 없이 계획만 출력",
    )

    args = parser.parse_args()

    create_indexes(
        qdrant_url=args.url,
        collection_name=args.collection,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
