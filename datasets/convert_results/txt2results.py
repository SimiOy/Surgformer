import numpy as np
import os
import glob

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("文件夹已创建：", folder_path)
    else:
        print("文件夹已存在：", folder_path)

Dataset2id = {
    "Cholec80": {"VIDEO_START_ID": 41, "VIDEO_END_ID": 80, "VIDEO_NAME_FORMAT": "video{id}"},
    "AutoLaparo": {"VIDEO_START_ID": 15, "VIDEO_END_ID": 21, "VIDEO_NAME_FORMAT": "video{id}"},
    "CATARACTS": {"VIDEO_START_ID": 1, "VIDEO_END_ID": 20, "VIDEO_NAME_FORMAT": "video{id}"},
    "PmLR50": {"VIDEO_START_ID": 1, "VIDEO_END_ID": 5, "VIDEO_NAME_FORMAT": "video{id}"},
    "M2CAI16-Workflow": {"VIDEO_START_ID": 28, "VIDEO_END_ID": 41, "VIDEO_NAME_FORMAT": "video_{id}"}
}
# 配置参数
Dataset = "M2CAI16-Workflow"  # 数据集名称
main_path = '/Users/yangshu/Downloads/论文投稿/SurgHybrid/results_final/M2CAI16-Workflow/ViT-B/Hybrid_B_M2CAI16_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4'

# 模式选择：True为自动映射模式，False为确定编号模式
AUTO_MAPPING_MODE = False

# 确定编号模式的参数（仅在AUTO_MAPPING_MODE=False时使用）
VIDEO_START_ID = Dataset2id[Dataset]["VIDEO_START_ID"]
VIDEO_END_ID = Dataset2id[Dataset]["VIDEO_END_ID"]
VIDEO_NAME_FORMAT = Dataset2id[Dataset]["VIDEO_NAME_FORMAT"]

# 自动查找所有数字命名的txt文件
txt_files = sorted(glob.glob(os.path.join(main_path, "[0-9]*.txt")))
print(f"找到 {len(txt_files)} 个txt文件: {[os.path.basename(f) for f in txt_files]}")

anns_path = main_path + "/phase_annotations"
pred_path = main_path + "/prediction"

create_folder_if_not_exists(anns_path)
create_folder_if_not_exists(pred_path)

# 读取所有txt文件的内容
all_lines = []
for txt_file in txt_files:
    with open(txt_file) as f:
        lines = f.readlines()
        all_lines.append(lines)

# 根据模式设置视频处理范围
if AUTO_MAPPING_MODE:
    # 自动映射模式：从所有文件中提取视频ID并创建映射
    video_id = set()
    for lines in all_lines:
        for i in lines[1:]:
            parts = i.split()
            if len(parts) > 1:
                video_name = parts[1]
                # 提取视频名称（去掉"video"前缀如果存在）
                if video_name.startswith('video_'):
                    video_name = video_name[6:]  # 去掉"video_"前缀
                elif video_name.startswith('video'):
                    video_name = video_name[5:]  # 去掉"video"前缀
                video_id.add(video_name)
    
    video_id = sorted(video_id)
    id_to_video = dict(zip(range(1, len(video_id) + 1), video_id))
    print("自动映射模式 - 视频ID映射：", id_to_video)
    video_range = range(1, len(video_id) + 1)
else:
    # 确定编号模式：使用指定的视频ID范围
    print(f"确定编号模式 - 处理视频ID: {VIDEO_START_ID} 到 {VIDEO_END_ID}")
    video_range = range(VIDEO_START_ID, VIDEO_END_ID + 1)
    id_to_video = None

# 生成phase_annotations文件
for i in video_range:
    if AUTO_MAPPING_MODE:
        video_name = id_to_video[i]
        output_filename = f"video-{i}.txt"
    else:
        video_name = str(i)
        output_filename = f"video-{i}.txt"
    
    # 使用字典收集当前视频的数据，自动去重
    video_data_dict = {}
    
    # 遍历所有txt文件
    for lines in all_lines:
        for j in range(1, len(lines)):
            temp = lines[j].split()
            file_video_name = temp[1]
            # 处理不同的视频名称格式
            expected_video_name = VIDEO_NAME_FORMAT.format(id=video_name)
            if file_video_name == expected_video_name or file_video_name == video_name:
                frame_num = int(temp[2])
                # M2CAI16 uses index 11 for phase, others use last element
                if Dataset == "M2CAI16-Workflow":
                    phase = int(temp[11])
                else:
                    phase = int(temp[-1])
                # 使用帧号作为key，后面的数据会覆盖前面的重复数据
                video_data_dict[frame_num] = phase
    
    # 转换为列表并按帧号排序
    video_data = [(frame_num, phase) for frame_num, phase in video_data_dict.items()]
    video_data.sort(key=lambda x: x[0])
    
    # 写入文件
    with open(os.path.join(anns_path, output_filename), "w") as f:
        f.write("Frame\tPhase\n")
        for frame_num, phase in video_data:
            f.write(f"{frame_num}\t{phase}\n")

# 生成prediction文件
for i in video_range:
    if AUTO_MAPPING_MODE:
        video_name = id_to_video[i]
        output_filename = f"video-{i}.txt"
        print(f"处理video-{i} (映射到: {video_name})")
    else:
        video_name = str(i)
        output_filename = f"video-{i}.txt"
        print(f"处理video-{i}")
    
    # 使用字典收集当前视频的预测数据，自动去重
    video_pred_data_dict = {}
    
    # 遍历所有txt文件
    for lines in all_lines:
        for j in range(1, len(lines)):
            temp_line = lines[j].strip()
            temp = lines[j].split()
            
            file_video_name = temp[1]
            # 处理不同的视频名称格式
            expected_video_name = VIDEO_NAME_FORMAT.format(id=video_name)
            if file_video_name == expected_video_name or file_video_name == video_name:
                try:
                    # 提取预测数据
                    data = np.fromstring(
                        temp_line.split("[")[1].split("]")[0], 
                        dtype=np.float32, 
                        sep=",",
                    )
                    predicted_phase = data.argmax()
                    frame_num = int(temp[2])
                    # 使用帧号作为key，后面的数据会覆盖前面的重复数据
                    video_pred_data_dict[frame_num] = predicted_phase
                except (IndexError, ValueError) as e:
                    print(f"警告: 处理第{j}行时出错: {e}")
                    continue
    
    # 转换为列表并按帧号排序
    video_pred_data = [(frame_num, predicted_phase) for frame_num, predicted_phase in video_pred_data_dict.items()]
    video_pred_data.sort(key=lambda x: x[0])
    
    # 写入文件
    with open(os.path.join(pred_path, output_filename), "w") as f:
        f.write("Frame\tPhase\n")
        for frame_num, predicted_phase in video_pred_data:
            f.write(f"{frame_num}\t{predicted_phase}\n")

print("处理完成！")
