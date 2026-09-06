#!/usr/bin/env python3
"""Stage2 수동 라벨링 도구 — entry_frame / evasion_space / entry_side

CCD(Car Crash Dataset) 사고 영상(5초·50프레임·10fps, 1,500건 로컬 확보 완료 —
crashvideo-project/DATA_SOURCES.md 1번 섹션 참고)을 직접 보면서, 대회가 요구하는
Stage2 라벨 4개 항목 중 CCD에 공식 라벨이 없는 3개(entry_frame, evasion_space,
entry_side)를 사람이 눈으로 보고 기록하는 로컬 GUI 도구다.

collision_frame은 새로 계산하지 않는다. CCD 공식 라벨 파일
data_raw/ccd/Crash-1500.txt 의 binlabels(50프레임 이진 라벨, 최초로 1이 되는
인덱스)를 그대로 파싱해 화면에 참고용으로만 띄운다 — 이 파싱 규칙은
DATA_SOURCES.md에서 1,500건 전수 검증을 마친 것과 동일하다(범위 30~49, 평균 37.2).

이 스크립트는 사용자가 자기 PC에서 직접 실행하고 키보드로 조작하는 대화형
프로그램이다. 에이전트가 대화형으로 조작할 수 없으므로, GUI 루프(main /
label_one_video)는 자동 테스트 대상이 아니고, 그 아래의 순수 함수들
(load_collision_frames / load_labeled_ids / list_pending_videos /
load_video_frames / append_result)만 헤드리스로 검증됐다.

## 준비물
    pip install opencv-python

## 실행
    # 저장소 루트에서 실행
    python src/label_stage2.py

## 조작키
    → 또는 d     다음 프레임        ← 또는 a     이전 프레임
    스페이스      현재 프레임을 entry_frame으로 기록
    1            entry_side = LEFT       2            entry_side = RIGHT
    y            evasion_space = 1(있음)  n            evasion_space = 0(없음)
    s            이 영상 건너뛰기(defer) — 라벨을 남기지 않고 다음 영상으로.
                 다음 실행 때 다시 나타난다 (완전히 skip 처리하고 싶으면 그냥
                 아무 값이나 채워서 저장하거나, 별도로 제외 목록을 관리할 것)
    q 또는 ESC   지금까지 작업 저장하고 종료

    entry_frame / entry_side / evasion_space 3개가 모두 채워지는 순간 자동으로
    해당 영상 라벨이 CSV에 append되고 다음 영상으로 넘어간다. 중간에 꺼도 이미
    완료된 영상들의 라벨은 남아있다. 재실행하면 이미 CSV에 있는 ID는 자동으로
    건너뛴다(resume).

## 출력
    crashvideo-project/data/stage2/labels_manual.csv
    컬럼: ID,collision_frame,entry_frame,evasion_space,entry_side
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path

from project_config import configured_path, load_project_properties

try:
    import cv2
except ImportError:  # pragma: no cover - 안내 메시지 목적
    print("opencv-python이 설치돼 있지 않습니다. `pip install opencv-python` 후 다시 실행하세요.",
          file=sys.stderr)
    raise

PROJECT_PROPERTIES = load_project_properties()
PROJECT_ROOT = Path(PROJECT_PROPERTIES["project.root"])
CCD_LABEL_FILE = configured_path("ccd.label.file", PROJECT_PROPERTIES)
CCD_VIDEO_DIR = configured_path("ccd.video.dir", PROJECT_PROPERTIES)
OUTPUT_CSV = configured_path("stage2.manual.labels", PROJECT_PROPERTIES)
OUTPUT_COLUMNS = ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"]

# data_raw/ccd/Crash-1500.txt 한 줄 형식:
#   000001,[0, 0, ..., 1, 1],000285,0000,Day,Normal,Yes
# (vidname, binlabels, startframe, youtubeID, timing, weather, egoinvolve)
# — DATA_SOURCES.md 1번 섹션에서 이미 검증된 것과 동일한 정규식.
CRASH_LINE_RE = re.compile(r"^(\d+),\[([^\]]*)\],(\d+),(\d+),(\w+),(\w+),(\w+)$")


# ---------------------------------------------------------------------------
# 순수 함수 (헤드리스로 테스트됨 — GUI 의존 없음)
# ---------------------------------------------------------------------------

def load_collision_frames(label_file: Path = CCD_LABEL_FILE) -> dict:
    """CCD 공식 라벨에서 영상별 collision_frame(binlabels 중 최초로 1이 되는
    인덱스, 0~49)을 파싱해 {video_id: collision_frame} 딕셔너리로 반환한다.
    새로 정의하는 로직이 아니라 이미 검증된 파싱 규칙을 그대로 재사용한다.
    """
    if not label_file.exists():
        raise FileNotFoundError(
            f"CCD 라벨 파일을 찾을 수 없습니다: {label_file}\n"
            "project.local.properties의 ccd.label.file 경로를 확인하세요"
            "(DATA_SOURCES.md 1번 섹션 참고)."
        )
    result = {}
    with open(label_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = CRASH_LINE_RE.match(line)
            if not m:
                continue
            vidname, binstr = m.group(1), m.group(2)
            binlabels = [int(x.strip()) for x in binstr.split(",")]
            collision_frame = binlabels.index(1) if 1 in binlabels else None
            result[vidname] = collision_frame
    return result


def load_labeled_ids(output_csv: Path = OUTPUT_CSV) -> set:
    """이미 labels_manual.csv에 기록된 영상 ID 집합 (resume에 사용)."""
    if not output_csv.exists():
        return set()
    ids = set()
    with open(output_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("ID"):
                ids.add(row["ID"])
    return ids


def list_pending_videos(video_dir: Path = CCD_VIDEO_DIR, collision_frames: dict = None,
                         labeled_ids: set = None) -> list:
    """라벨링 대상 영상 ID 목록(정렬됨). 이미 라벨링된 것과, 라벨은 있지만
    실제 mp4 파일이 없는 것은 제외한다."""
    if collision_frames is None:
        collision_frames = load_collision_frames()
    if labeled_ids is None:
        labeled_ids = load_labeled_ids()
    pending = []
    for vid in sorted(collision_frames.keys()):
        if vid in labeled_ids:
            continue
        if (video_dir / f"{vid}.mp4").exists():
            pending.append(vid)
    return pending


def load_video_frames(video_id: str, video_dir: Path = CCD_VIDEO_DIR) -> list:
    """영상 하나를 전부 읽어 BGR 프레임 리스트로 반환한다. CCD 영상은 50프레임
    (~수백KB)이라 전체 로드해도 가볍고, 매 프레임 seek보다 안정적이다."""
    path = video_dir / f"{video_id}.mp4"
    cap = cv2.VideoCapture(str(path))
    frames = []
    if cap.isOpened():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    cap.release()
    return frames


def append_result(row: dict, output_csv: Path = OUTPUT_CSV) -> None:
    """영상 한 건의 라벨을 CSV에 즉시 append(+flush)한다 — 중간에 꺼져도 이미
    완료된 행은 보존된다."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not output_csv.exists()
    with open(output_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# GUI (대화형 — 로컬에서 사람이 직접 실행/조작. 자동 테스트 대상 아님)
# ---------------------------------------------------------------------------

WINDOW_NAME = "Stage2 Manual Labeling (CCD)"

# cv2.waitKeyEx() 확장 코드. 화살표 키는 OS마다 값이 달라 Windows 기준값을
# 우선 넣었고(이 프로젝트 환경), a/d는 플랫폼 무관하게 항상 동작하는
# 주 조작키다 — 화살표가 안 먹으면 a/d를 쓰면 된다.
KEY_LEFT_EXTRA = {2424832, 65361, 63234}   # Windows / Linux / macOS 등 Left
KEY_RIGHT_EXTRA = {2555904, 65363, 63235}  # Windows / Linux / macOS 등 Right

KEY_LEGEND = [
    "[a/LEFT]prev  [d/RIGHT]next   [SPACE]mark entry_frame",
    "[1]entry_side=LEFT  [2]entry_side=RIGHT   [y/n]evasion_space",
    "[s]skip(defer to next run)   [q/ESC]save & quit",
]


def draw_overlay(frame, video_id, frame_idx, num_frames, collision_frame,
                  entry_frame, entry_side, evasion_space):
    img = frame.copy()
    h, w = img.shape[:2]

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 95), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    def put(text, y, color=(255, 255, 255)):
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    put(f"ID={video_id}   frame {frame_idx}/{num_frames - 1}   "
        f"collision_frame(ref)={collision_frame}", 20)

    ef = entry_frame if entry_frame is not None else "-"
    es = entry_side if entry_side is not None else "-"
    ev = evasion_space if evasion_space is not None else "-"
    marker = "  <- entry_frame" if entry_frame == frame_idx else ""
    marker += "  <- collision_frame" if collision_frame == frame_idx else ""
    put(f"entry_frame={ef}{marker}", 42,
        color=(0, 255, 0) if entry_frame is not None else (0, 255, 255))
    put(f"entry_side={es}    evasion_space={ev}", 64,
        color=(0, 255, 0) if (entry_side is not None and evasion_space is not None) else (0, 255, 255))
    put(KEY_LEGEND[0], 86, (200, 200, 200))

    cv2.rectangle(img, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(img, KEY_LEGEND[1], (10, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, KEY_LEGEND[2], (10, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def label_one_video(video_id, frames, collision_frame):
    """영상 한 건을 라벨링한다.
    반환값: dict(완료된 라벨 행) / None(skip) / "QUIT"(전체 종료 요청)."""
    num_frames = len(frames)
    idx = collision_frame if collision_frame is not None else 0
    idx = max(0, min(num_frames - 1, idx))  # collision_frame 근처에서 시작하면 탐색이 빠름
    entry_frame = None
    entry_side = None
    evasion_space = None

    while True:
        img = draw_overlay(frames[idx], video_id, idx, num_frames, collision_frame,
                            entry_frame, entry_side, evasion_space)
        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKeyEx(0)
        key_low = key & 0xFF

        if key in KEY_LEFT_EXTRA or key_low == ord('a'):
            idx = max(0, idx - 1)
        elif key in KEY_RIGHT_EXTRA or key_low == ord('d'):
            idx = min(num_frames - 1, idx + 1)
        elif key_low == ord(' '):
            entry_frame = idx
        elif key_low == ord('1'):
            entry_side = "LEFT"
        elif key_low == ord('2'):
            entry_side = "RIGHT"
        elif key_low == ord('y'):
            evasion_space = 1
        elif key_low == ord('n'):
            evasion_space = 0
        elif key_low == ord('s'):
            return None
        elif key_low in (ord('q'), 27):  # ESC
            return "QUIT"

        if entry_frame is not None and entry_side is not None and evasion_space is not None:
            return {
                "ID": video_id,
                "collision_frame": collision_frame,
                "entry_frame": entry_frame,
                "evasion_space": evasion_space,
                "entry_side": entry_side,
            }


def main():
    parser = argparse.ArgumentParser(description="CCD Stage2 manual labeling")
    parser.add_argument("--start-id", type=int, default=1,
                        help="Start at this video ID; earlier pending videos remain available next run")
    args = parser.parse_args()
    collision_frames = load_collision_frames()
    labeled_ids = load_labeled_ids()
    pending = list_pending_videos(collision_frames=collision_frames, labeled_ids=labeled_ids)
    pending = [video_id for video_id in pending if int(video_id) >= args.start_id]

    if not pending:
        print("라벨링할 영상이 없습니다 — 이미 전부 완료됐거나 "
              "data_raw/ccd/videos/Crash-1500 아래에 mp4가 없습니다.")
        return

    print(f"CCD {len(collision_frames)}건 중 이미 완료 {len(labeled_ids)}건 제외, "
          f"이번 세션 대상 {len(pending)}건.")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    done = 0
    try:
        for video_id in pending:
            frames = load_video_frames(video_id)
            if not frames:
                print(f"[경고] {video_id}: 프레임을 읽지 못해 건너뜁니다.")
                continue
            collision_frame = collision_frames.get(video_id)
            result = label_one_video(video_id, frames, collision_frame)

            if result == "QUIT":
                print(f"\n종료합니다. 이번 세션 라벨링 {done}건 저장 완료.")
                break
            elif result is None:
                print(f"[skip] {video_id} (다음 실행 때 다시 표시됩니다)")
                continue
            else:
                append_result(result)
                done += 1
                print(f"[저장 {done}] {video_id} -> {result}")
    finally:
        cv2.destroyAllWindows()

    print(f"완료. 이번 세션에서 {done}건을 {OUTPUT_CSV}에 저장했습니다.")


if __name__ == "__main__":
    main()
