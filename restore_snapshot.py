#!/usr/bin/env python3
"""
Qdrant Snapshot Restore Script

Usage:
    # 로컬 파일에서 복원 (Qdrant 서버가 접근 가능한 경로)
    python restore_snapshot.py --file /path/to/snapshot.snapshot

    # HTTP URL에서 복원
    python restore_snapshot.py --url http://server/snapshots/snapshot.snapshot

    # 업로드하여 복원 (로컬 파일을 Qdrant에 업로드)
    python restore_snapshot.py --upload ./my_snapshot.snapshot

References:
    - https://qdrant.tech/documentation/concepts/snapshots/
    - https://api.qdrant.tech/api-reference/snapshots/recover-from-snapshot
"""

import argparse
import sys
from pathlib import Path

import httpx
from qdrant_client import QdrantClient, models


def get_client(url: str, api_key: str | None = None) -> QdrantClient:
    """Qdrant 클라이언트 생성"""
    return QdrantClient(url=url, api_key=api_key)


def recover_from_file(
    client: QdrantClient,
    collection_name: str,
    file_path: str,
    priority: str = "snapshot",
) -> bool:
    """
    Qdrant 서버가 접근 가능한 로컬 파일에서 복원

    Note: file_path는 Qdrant 서버 기준 경로여야 합니다.
          Docker 환경에서는 볼륨 마운트된 경로를 사용하세요.
    """
    # file:// URI 형식으로 변환
    if not file_path.startswith("file://"):
        file_path = f"file://{file_path}"

    print(f"Recovering collection '{collection_name}' from: {file_path}")

    priority_map = {
        "snapshot": models.SnapshotPriority.SNAPSHOT,
        "replica": models.SnapshotPriority.REPLICA,
        "no_sync": models.SnapshotPriority.NO_SYNC,
    }

    client.recover_snapshot(
        collection_name=collection_name,
        location=file_path,
        priority=priority_map.get(priority, models.SnapshotPriority.SNAPSHOT),
        wait=True,
    )

    print(f"Successfully recovered collection '{collection_name}'")
    return True


def recover_from_url(
    client: QdrantClient,
    collection_name: str,
    url: str,
    priority: str = "snapshot",
) -> bool:
    """HTTP/HTTPS URL에서 스냅샷 복원"""
    print(f"Recovering collection '{collection_name}' from URL: {url}")

    priority_map = {
        "snapshot": models.SnapshotPriority.SNAPSHOT,
        "replica": models.SnapshotPriority.REPLICA,
        "no_sync": models.SnapshotPriority.NO_SYNC,
    }

    client.recover_snapshot(
        collection_name=collection_name,
        location=url,
        priority=priority_map.get(priority, models.SnapshotPriority.SNAPSHOT),
        wait=True,
    )

    print(f"Successfully recovered collection '{collection_name}'")
    return True


def recover_from_upload(
    qdrant_url: str,
    collection_name: str,
    local_file: str,
    priority: str = "snapshot",
    api_key: str | None = None,
) -> bool:
    """
    로컬 파일을 Qdrant에 업로드하여 복원

    이 방식은 Qdrant 서버가 로컬 파일에 직접 접근할 수 없을 때 사용합니다.
    파일이 HTTP POST를 통해 업로드됩니다.

    Reference: https://api.qdrant.tech/api-reference/snapshots/recover-from-uploaded-snapshot
    """
    file_path = Path(local_file)
    if not file_path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {local_file}")

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"Uploading snapshot ({file_size_mb:.1f} MB) to Qdrant...")
    print(f"Collection: {collection_name}")
    print(f"Priority: {priority}")
    print("")

    # REST API 엔드포인트
    url = f"{qdrant_url.rstrip('/')}/collections/{collection_name}/snapshots/upload"
    params = {"priority": priority, "wait": "true"}

    # 헤더 설정
    headers = {}
    if api_key:
        headers["api-key"] = api_key

    # 파일 업로드 (httpx)
    print("Uploading... (this may take a while for large files)")
    with open(file_path, "rb") as f:
        files = {"snapshot": (file_path.name, f, "application/octet-stream")}
        # httpx는 대용량 파일 업로드 시 timeout 설정 필요
        timeout = httpx.Timeout(timeout=3600.0, connect=30.0)
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, params=params, headers=headers, files=files)

    # 응답 확인
    if response.status_code == 200:
        result = response.json()
        print(f"Successfully recovered collection '{collection_name}'")
        print(f"Result: {result}")
        return True
    else:
        raise RuntimeError(f"Failed to upload snapshot: {response.status_code} - {response.text}")


def list_collections(client: QdrantClient) -> list[str]:
    """모든 컬렉션 목록 조회"""
    collections = client.get_collections()
    return [c.name for c in collections.collections]


def get_collection_info(client: QdrantClient, collection_name: str) -> dict:
    """컬렉션 정보 조회"""
    try:
        info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status,
        }
    except Exception:
        return {"name": collection_name, "exists": False}


def main():
    parser = argparse.ArgumentParser(
        description="Qdrant Snapshot Restore Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 서버 로컬 파일에서 복원 (Qdrant가 접근 가능한 경로)
  python restore_snapshot.py --file /qdrant/snapshots/my_snapshot.snapshot

  # URL에서 복원
  python restore_snapshot.py --url http://backup-server/snapshots/my_snapshot.snapshot

  # 로컬 파일 업로드하여 복원 (권장)
  python restore_snapshot.py --upload ./my_snapshot.snapshot

  # 컬렉션 목록 확인
  python restore_snapshot.py --list

  # 특정 컬렉션 정보 확인
  python restore_snapshot.py --info
        """,
    )

    # Qdrant 연결 설정
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant server URL (default: http://localhost:6333)",
    )
    parser.add_argument("--api-key", help="Qdrant API key (if required)")
    parser.add_argument(
        "--collection",
        default="naver_connect_docs",
        help="Collection name (default: naver_connect_docs)",
    )

    # 복원 방식 (상호 배타적)
    restore_group = parser.add_mutually_exclusive_group()
    restore_group.add_argument(
        "--file",
        metavar="PATH",
        help="Restore from file path (Qdrant server must have access)",
    )
    restore_group.add_argument("--url", metavar="URL", help="Restore from HTTP/HTTPS URL")
    restore_group.add_argument(
        "--upload",
        metavar="FILE",
        help="Upload local file and restore (recommended for remote Qdrant)",
    )
    restore_group.add_argument("--list", action="store_true", help="List all collections")
    restore_group.add_argument("--info", action="store_true", help="Show collection info")

    # 추가 옵션
    parser.add_argument(
        "--priority",
        choices=["snapshot", "replica", "no_sync"],
        default="snapshot",
        help="Recovery priority (default: snapshot)",
    )

    args = parser.parse_args()

    # 클라이언트 생성
    try:
        client = get_client(args.qdrant_url, args.api_key)
        print(f"Connected to Qdrant at {args.qdrant_url}")
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}", file=sys.stderr)
        sys.exit(1)

    # 명령 실행
    try:
        if args.list:
            collections = list_collections(client)
            print("\nCollections:")
            for name in collections:
                print(f"  - {name}")
            if not collections:
                print("  (no collections)")

        elif args.info:
            info = get_collection_info(client, args.collection)
            print(f"\nCollection Info: {args.collection}")
            for key, value in info.items():
                print(f"  {key}: {value}")

        elif args.file:
            recover_from_file(client, args.collection, args.file, args.priority)

        elif args.url:
            recover_from_url(client, args.collection, args.url, args.priority)

        elif args.upload:
            recover_from_upload(args.qdrant_url, args.collection, args.upload, args.priority, args.api_key)

        else:
            parser.print_help()
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
