"""
Streaming Co3D Dataset Processor (parallel + resume)
- 카테고리별로: download → preprocess → test 이미지만 (트리 보존) 복사 → zip/원본 삭제
- 이미 처리된 카테고리는 건너뛰기
- 최대 5개 병렬 처리
"""

import os
import shutil
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== 설정 =====
MAX_WORKERS = 5  # 병렬 개수
DOWNLOAD_TIMEOUT_SEC = 3600
EXTRACT_WAIT_MAX_SEC = 300

# 기본 경로 (원하면 수정)
DEFAULT_DOWNLOAD_DIR = "co3d_streaming_temp"
DEFAULT_OUTPUT_DIR = "/workspace/toddler/vggt/co3d_annotations_full"

# Co3D 카테고리 목록
CATEGORIES = [
    "apple", "backpack", "ball", "banana", "baseballbat", "baseballglove",
    "bench", "bicycle", "book", "bottle", "bowl", "broccoli", "cake", "car", "carrot",
    "cellphone", "chair", "couch", "cup", "donut", "frisbee", "hairdryer", "handbag",
    "hotdog", "hydrant", "keyboard", "kite", "laptop", "microwave", "motorcycle",
    "mouse", "orange", "parkingmeter", "pizza", "plant", "remote", "sandwich",
    "skateboard", "stopsign", "suitcase", "teddybear", "toaster", "toilet", "toybus",
    "toyplane", "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass",
]

# ---------- 유틸 ----------
def _exists_all_test_images(category: str, output_dir: str) -> bool:
    """annotation의 filepath 전부가 output_dir/images/<filepath>에 존재하는지 체크"""
    import gzip, json
    test_jgz = Path(output_dir) / f"{category}_test.jgz"
    train_jgz = Path(output_dir) / f"{category}_train.jgz"
    if not test_jgz.exists() or not train_jgz.exists():
        return False

    try:
        with gzip.open(test_jgz, "r") as fin:
            data = json.loads(fin.read())
    except Exception:
        return False

    images_root = Path(output_dir) / "images"
    # 테스트 프레임 총 개수
    filepaths = []
    for seq_frames in data.values():
        for fr in seq_frames:
            fp = fr["filepath"]  # 예: "category/sequence/images/xxx.jpg"
            filepaths.append(fp)

    if not filepaths:
        return False

    # 모두 존재하는지 확인 (빠른 실패)
    for rel in filepaths:
        if not (images_root / Path(*rel.split("/"))).exists():
            return False
    return True


def is_category_done(category: str, output_dir: str) -> bool:
    """카테고리 처리 완료 여부 판단"""
    done = _exists_all_test_images(category, output_dir)
    if done:
        print(f"⏭️  이미 처리됨: {category}")
    return done


def download_category(category: str, download_dir: str) -> bool:
    """특정 카테고리 다운로드 및 압축 해제 완료 대기"""
    print(f"📥 다운로드 중: {category}")
    cmd = [
        "python", "co3d/co3d/download_dataset.py",
        "--download_folder", download_dir,
        "--download_categories", category,
    ]
    try:
        result = subprocess.run(cmd, timeout=DOWNLOAD_TIMEOUT_SEC)
        if result.returncode != 0:
            print(f"❌ {category} 다운로드 실패 (returncode={result.returncode})")
            return False
        print(f"✅ {category} 다운로드 완료")

        # 압축 해제 대기
        category_path = Path(download_dir) / category
        waited = 0
        while waited < EXTRACT_WAIT_MAX_SEC:
            if category_path.exists() and any(category_path.iterdir()):
                print(f"✅ {category} 압축 해제 완료")
                return True
            time.sleep(5)
            waited += 5
            print(f"⏳ {category} 압축 해제 대기 중... ({waited}s)")
        print(f"⚠️ {category} 압축 해제 타임아웃")
        return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {category} 다운로드 타임아웃")
        return False


def preprocess_category(category: str, co3d_dir: str, output_dir: str) -> bool:
    """원본 전처리 스크립트 호출"""
    print(f"🔄 전처리 중: {category}")
    cmd = [
        "python", "evaluation/preprocess_co3d.py",
        "--category", category,
        "--co3d_v2_dir", co3d_dir,
        "--output_dir", output_dir,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {category} 전처리 완료")
            return True
        else:
            print(f"❌ {category} 전처리 실패:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {category} 전처리 오류: {e}")
        return False


def remove_category_zips(download_dir: str, category: str) -> int:
    """해당 카테고리 관련 zip 파일 제거"""
    removed = 0
    root = Path(download_dir)
    for z in root.rglob("*.zip"):
        name = z.name.lower()
        parts_lower = [p.lower() for p in z.parts]
        if (category.lower() in parts_lower) or \
           name.startswith(category.lower()) or \
           f"_{category.lower()}_" in name or \
           name == f"{category.lower()}.zip":
            try:
                z.unlink()
                removed += 1
            except Exception as e:
                print(f"⚠️ zip 삭제 실패: {z} ({e})")
    if removed > 0:
        print(f"🧹 {category} 관련 zip {removed}개 삭제")
    return removed


def cleanup_category(category: str, download_dir: str, output_dir: str) -> bool:
    """test set 이미지만 (트리 보존) 복사 후, 원본 카테고리 폴더 삭제"""
    print(f"🗑️ 정리 중: {category}")
    import gzip, json

    test_annotation_file = Path(output_dir) / f"{category}_test.jgz"
    if not test_annotation_file.exists():
        print(f"⚠️ {category} test annotation 없음: {test_annotation_file}")
    else:
        with gzip.open(test_annotation_file, "r") as fin:
            test_annotation = json.loads(fin.read())

        # 복사 대상
        test_image_paths = set()
        for seq_frames in test_annotation.values():
            for frame in seq_frames:
                test_image_paths.add(frame["filepath"])  # "category/seq/images/xxx.jpg"

        print(f"📊 {category} test 이미지: {len(test_image_paths)}")

        image_root = Path(output_dir) / "images"
        copied = 0
        for img_rel in sorted(test_image_paths):
            dst_path = image_root / Path(*img_rel.split("/"))
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            parts = img_rel.split("/")
            if len(parts) >= 2 and parts[0] == category:
                src_path = Path(download_dir) / Path(*parts)
            else:
                src_path = Path(download_dir) / category / Path(*parts)

            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                copied += 1
            else:
                print(f"⚠️ 없음: {src_path}")

        print(f"✅ {category} 복사 완료 (디렉토리 구조 유지, {copied}개)")

    # 원본 카테고리 폴더 삭제
    category_path = Path(download_dir) / category
    if category_path.exists():
        shutil.rmtree(category_path)
        print(f"✅ {category} 원본 폴더 삭제 완료")

    return True


def process_category_streaming(category: str, download_dir: str, output_dir: str) -> str:
    """
    카테고리 하나 처리 파이프라인.
    반환: "done" | "skipped" | "failed"
    """
    try:
        if is_category_done(category, output_dir):
            return "skipped"

        # 1) 다운로드
        if not download_category(category, download_dir):
            return "failed"

        # 2) 전처리
        if not preprocess_category(category, download_dir, output_dir):
            return "failed"

        # 3) zip 삭제
        remove_category_zips(download_dir, category)

        # 4) test 이미지만 복사 + 원본 폴더 삭제
        if not cleanup_category(category, download_dir, output_dir):
            return "failed"

        # 최종 검증 (스킵 기준과 동일)
        if not _exists_all_test_images(category, output_dir):
            print(f"⚠️ {category} 최종 검증 실패(이미지 누락)")
            return "failed"

        print(f"🎉 {category} 처리 완료!")
        return "done"

    except Exception as e:
        print(f"💥 {category} 처리 중 예외: {e}")
        return "failed"


def main():
    download_dir = DEFAULT_DOWNLOAD_DIR
    output_dir = DEFAULT_OUTPUT_DIR

    Path(download_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("🚀 Co3D 스트리밍 처리 시작!")
    print(f"📁 임시 다운로드: {download_dir}")
    print(f"📁 출력: {output_dir}")
    print(f"📊 총 카테고리: {len(CATEGORIES)}개")

    # 이미 처리된 것은 미리 거르기
    pending = [c for c in CATEGORIES if not is_category_done(c, output_dir)]
    print(f"📝 처리 대상: {len(pending)}개 (스킵 {len(CATEGORIES) - len(pending)})")

    results = {"done": [], "skipped": [], "failed": []}

    # 병렬 처리
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_map = {ex.submit(process_category_streaming, c, download_dir, output_dir): c for c in pending}
        for fut in as_completed(future_map):
            cat = future_map[fut]
            try:
                status = fut.result()
            except Exception as e:
                print(f"💥 {cat} future 예외: {e}")
                status = "failed"
            results.setdefault(status, []).append(cat)

    # 요약 출력
    print("\n" + "=" * 60)
    print("🏁 병렬 처리 완료 요약")
    print(f"✅ 완료: {len(results['done'])} → {results['done']}")
    print(f"⏭️ 스킵: {len(results['skipped'])} → {results['skipped']}")
    print(f"❌ 실패: {len(results['failed'])} → {results['failed']}")

    # (선택) 남아있는 zip 전체 정리
    total_removed = 0
    for z in Path(download_dir).rglob("*.zip"):
        try:
            z.unlink()
            total_removed += 1
        except Exception as e:
            print(f"⚠️ 잔여 zip 삭제 실패: {z} ({e})")
    if total_removed > 0:
        print(f"🧹 잔여 zip {total_removed}개 삭제 완료")

    # 임시 디렉토리 정리
    if Path(download_dir).exists():
        try:
            shutil.rmtree(download_dir)
            print(f"🗑️ 임시 디렉토리 정리 완료")
        except Exception as e:
            print(f"⚠️ 임시 디렉토리 정리 실패: {e}")

    print(f"📁 최종 annotation 및 이미지 루트: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
