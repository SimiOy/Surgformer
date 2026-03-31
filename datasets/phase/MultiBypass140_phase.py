import os
import numpy as np
import h5py
from PIL import Image

from datasets.phase.Cholec80_phase import PhaseDataset_Cholec80


class PhaseDataset_MultiBypass140(PhaseDataset_Cholec80):
    """MultiBypass140 phase dataset (HDF5 version).

    Frames are stored in per-video HDF5 files:

      {data_path}/frames_hdf5/{video_id}.h5
        /frames            (N, H, W, 3) uint8 RGB
        /frames_cutmargin  (N, 250, 250, 3) uint8 RGB

    Labels are loaded from converted pickle files:

      {data_path}/labels/train/1fpstrain.pickle
      {data_path}/labels/val/1fpsval.pickle
      {data_path}/labels/test/1fpstest.pickle
    """

    def __init__(self, *args, **kwargs):
        self._h5_handles = {}
        super().__init__(*args, **kwargs)

    def _make_dataset(self, infos):
        frames = []
        for video_id in infos.keys():
            for line_info in infos[video_id]:
                # keep the same expectation style as Cholec80/M2CAI16
                required_keys = ["video_id", "frame_id", "phase_gt", "frames"]
                for k in required_keys:
                    if k not in line_info:
                        raise RuntimeError(
                            f"Missing required key '{k}' in sample: {line_info}"
                        )

                hdf5_path = os.path.join(
                    self.data_path,
                    "frames_hdf5",
                    line_info["video_id"] + ".h5",
                )
                line_info["hdf5_path"] = hdf5_path
                line_info["img_path"] = hdf5_path  # kept for error messages / compatibility
                frames.append(line_info)
        return frames

    def _read_frame(self, image_index, cut_black):
        """Return a PIL Image for the given dataset sample index."""
        sample = self.dataset_samples[image_index]
        hdf5_path = sample["hdf5_path"]
        frame_id = sample["frame_id"]
        key = "frames_cutmargin" if cut_black else "frames"

        if hdf5_path not in self._h5_handles:
            self._h5_handles[hdf5_path] = h5py.File(hdf5_path, "r")

        arr = self._h5_handles[hdf5_path][key][frame_id]  # (H, W, 3), uint8 RGB
        return Image.fromarray(arr)

    def _video_batch_loader(self, duration, indice, video_id, index, cut_black):
        offset_value = index - indice
        frame_sample_rate = self.frame_sample_rate
        frame_id_list = []

        import random

        for i, _ in enumerate(range(0, self.clip_len)):
            frame_id_list.append(indice)
            if self.frame_sample_rate == -1:
                frame_sample_rate = random.randint(1, 5)
            elif self.frame_sample_rate == 0:
                frame_sample_rate = 2 ** i
            elif self.frame_sample_rate == -2:
                frame_sample_rate = 1 if 2 * i == 0 else 2 * i

            if indice - frame_sample_rate >= 0:
                indice -= frame_sample_rate

        sampled_list = sorted([i + offset_value for i in frame_id_list])

        sampled_image_list = []
        sampled_label_list = []

        for num, image_index in enumerate(sampled_list):
            try:
                image_data = self._read_frame(image_index, cut_black)
                phase_label = self.dataset_samples[image_index]["phase_gt"]
                sampled_image_list.append(image_data)
                sampled_label_list.append(phase_label)
            except Exception:
                raise RuntimeError(
                    "Error reading frame {} from video {} (index {}).".format(
                        frame_id_list[num], video_id, image_index
                    )
                )

        return (
            np.stack(sampled_image_list),
            np.stack(sampled_label_list),
            sampled_list,
        )

    def _video_batch_loader_for_key_frames(
        self, duration, timestamp, video_id, index, cut_black
    ):
        import random

        right_len = self.clip_len // 2
        left_len = self.clip_len - right_len
        offset_value = index - timestamp

        # load right side
        right_sample_rate = self.frame_sample_rate
        cur_t = timestamp
        right_frames = []

        if right_len == left_len:
            for i, _ in enumerate(range(0, right_len)):
                right_frames.append(cur_t)
                if self.frame_sample_rate == -1:
                    right_sample_rate = random.randint(1, 5)
                elif self.frame_sample_rate == 0:
                    right_sample_rate = 2 ** i
                elif self.frame_sample_rate == -2:
                    right_sample_rate = 1 if 2 * i == 0 else 2 * i
                if cur_t + right_sample_rate <= duration:
                    cur_t += right_sample_rate
        else:
            for i, _ in enumerate(range(0, right_len)):
                if self.frame_sample_rate == -1:
                    right_sample_rate = random.randint(1, 5)
                elif self.frame_sample_rate == 0:
                    right_sample_rate = 2 ** i
                elif self.frame_sample_rate == -2:
                    right_sample_rate = 1 if 2 * i == 0 else 2 * i
                if cur_t + right_sample_rate <= duration:
                    cur_t += right_sample_rate
                right_frames.append(cur_t)

        # load left side
        left_sample_rate = self.frame_sample_rate
        cur_t = timestamp
        left_frames = []

        for j, _ in enumerate(range(0, left_len)):
            left_frames = [cur_t] + left_frames
            if self.frame_sample_rate == -1:
                left_sample_rate = random.randint(1, 5)
            elif self.frame_sample_rate == 0:
                left_sample_rate = 2 ** j
            elif self.frame_sample_rate == -2:
                left_sample_rate = 1 if 2 * j == 0 else 2 * j
            if cur_t - left_sample_rate >= 0:
                cur_t -= left_sample_rate

        frame_id_list = left_frames + right_frames
        assert len(frame_id_list) == self.clip_len
        sampled_list = [i + offset_value for i in frame_id_list]

        sampled_image_list = []
        sampled_label_list = []

        for num, image_index in enumerate(sampled_list):
            try:
                image_data = self._read_frame(image_index, cut_black)
                phase_label = self.dataset_samples[image_index]["phase_gt"]
                sampled_image_list.append(image_data)
                sampled_label_list.append(phase_label)
            except Exception:
                raise RuntimeError(
                    "Error reading frame {} from video {} (index {}).".format(
                        frame_id_list[num], video_id, image_index
                    )
                )

        return (
            np.stack(sampled_image_list),
            np.stack(sampled_label_list),
            sampled_list,
        )