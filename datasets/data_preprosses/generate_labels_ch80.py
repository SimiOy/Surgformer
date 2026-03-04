import os
import pickle
from tqdm import tqdm


def main():
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--data-dir", default="data/Cholec80",
                    help="Root Cholec80 directory (TF-Cholec80 format, no raw videos)")
    _args, _ = _p.parse_known_args()
    ROOT_DIR = _args.data_dir

    phase2id = {
        'Preparation': 0, 'CalotTriangleDissection': 1, 'ClippingCutting': 2,
        'GallbladderDissection': 3, 'GallbladderPackaging': 4,
        'CleaningCoagulation': 5, 'GallbladderRetraction': 6,
    }

    TRAIN_NUMBERS = set(range(1, 41))
    TEST_NUMBERS = set(range(41, 81))

    train_pkl = dict()
    test_pkl = dict()
    unique_id_train = 0
    unique_id_test = 0

    phase_ann_dir = os.path.join(ROOT_DIR, 'phase_annotations')
    tool_ann_dir = os.path.join(ROOT_DIR, 'tool_annotations')
    frames_dir = os.path.join(ROOT_DIR, 'frames')

    ann_files = sorted(f for f in os.listdir(phase_ann_dir) if f.endswith('-phase.txt'))

    for ann_file in tqdm(ann_files):
        video_id = ann_file.replace('-phase.txt', '')   # e.g. "video03"
        vid_num = int(video_id.replace('video', ''))

        if not os.path.isdir(os.path.join(frames_dir, video_id)):
            print(f"Skipping {video_id}: no frames directory found")
            continue

        # Phase annotations: 1fps, header "Frame\tPhase", entries "0\tPreparation" etc.
        phase_path = os.path.join(phase_ann_dir, ann_file)
        with open(phase_path, 'r') as f:
            phase_lines = f.readlines()[1:]  # skip header

        # Tool annotations: 25fps frame index (0, 25, 50 ... = seconds 0, 1, 2 ...)
        tool_dict = {}
        tool_path = os.path.join(tool_ann_dir, video_id + '-tool.txt')
        if os.path.exists(tool_path):
            with open(tool_path, 'r') as f:
                f.readline()  # skip header
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        tool_dict[parts[0]] = list(map(int, parts[1:]))

        total_frames = len(phase_lines)

        if vid_num in TRAIN_NUMBERS:
            unique_id = unique_id_train
        else:
            unique_id = unique_id_test

        frame_infos = []
        for line in phase_lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            frame_id_1fps = int(parts[0])   # 0-indexed 1fps frame number
            phase_name = parts[1]

            tool_gt = tool_dict.get(str(frame_id_1fps * 25), None)

            info = {
                'unique_id': unique_id,
                'frame_id': frame_id_1fps,      # 0-indexed; filename = videoXX_XXXXXX.png where index = frame_id+1
                'video_id': video_id,
                'tool_gt': tool_gt,
                'phase_gt': phase2id[phase_name],
                'phase_name': phase_name,
                'fps': 1,
                'frames': total_frames,
            }
            frame_infos.append(info)
            unique_id += 1

        if vid_num in TRAIN_NUMBERS:
            train_pkl[video_id] = frame_infos
            unique_id_train = unique_id
        elif vid_num in TEST_NUMBERS:
            test_pkl[video_id] = frame_infos
            unique_id_test = unique_id

    train_save_dir = os.path.join(ROOT_DIR, 'labels', 'train')
    os.makedirs(train_save_dir, exist_ok=True)
    with open(os.path.join(train_save_dir, '1fpstrain.pickle'), 'wb') as f:
        pickle.dump(train_pkl, f)

    test_save_dir = os.path.join(ROOT_DIR, 'labels', 'test')
    os.makedirs(test_save_dir, exist_ok=True)
    with open(os.path.join(test_save_dir, '1fpsval_test.pickle'), 'wb') as f:
        pickle.dump(test_pkl, f)

    print(f'Train videos: {len(train_pkl)}, frames: {unique_id_train}')
    print(f'Test  videos: {len(test_pkl)}, frames: {unique_id_test}')


if __name__ == '__main__':
    main()