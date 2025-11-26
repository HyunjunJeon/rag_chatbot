"""
Qdrant 컬렉션 조회 스크립트.
기존 컬렉션들의 상태와 문서 수를 확인합니다.
"""

from qdrant_client import QdrantClient


def main() -> None:
    """모든 Qdrant 컬렉션 정보를 조회합니다."""
    qdrant_url = "http://localhost:6333"

    print("=" * 80)
    print("🔍 Qdrant 컬렉션 조회")
    print("=" * 80)

    client = QdrantClient(url=qdrant_url)

    # 모든 컬렉션 조회
    collections = client.get_collections().collections
    print(f"\n📦 총 컬렉션 수: {len(collections)}")

    if not collections:
        print("   컬렉션이 없습니다.")
        return

    print("\n" + "-" * 80)

    for coll in sorted(collections, key=lambda x: x.name):
        try:
            info = client.get_collection(coll.name)
            print(f"\n📊 {coll.name}")
            print(f"   문서 수: {info.points_count:,}개")
            print(f"   벡터 차원: {info.config.params.vectors.size}")
            print(f"   상태: {info.status}")

            # 샘플 문서 조회 (1개)
            sample = client.scroll(
                collection_name=coll.name,
                limit=1,
                with_payload=True,
            )
            if sample[0]:
                payload = sample[0][0].payload
                print(f"   doc_type: {payload.get('doc_type', 'N/A')}")
                print(f"   course 예시: {payload.get('course', 'N/A')}")

        except Exception as e:
            print(f"\n📊 {coll.name}")
            print(f"   ✗ 조회 실패: {e}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
