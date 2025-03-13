#!/usr/bin/env python
import os
import shutil
import numpy as np
import zarr
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def convert_zarr_to_lerobot(zarr_path, output_repo, fps=30, task_description="Grasp and place an object.", push_to_hub=False):
    """
    将Zarr数据转换为LeRobot格式
    
    Args:
        zarr_path: Zarr数据的路径
        output_repo: 输出数据集的仓库名称"
        fps: 数据集的帧率，默认为30
        task_description: 任务描述，默认为"Grasp and place an object."
        push_to_hub: 是否将数据集推送到Hugging Face Hub
    """
    print(f"正在打开Zarr数据: {zarr_path}")
    zarr_data = zarr.open(zarr_path, mode='r')
    
    # 检查必要的数据是否存在
    required_paths = ['data/left_wrist_img', 'data/left_robot_tcp_pose', 'data/action']
    for path in required_paths:
        if path not in zarr_data:
            raise ValueError(f"Zarr数据中缺少必要的路径: {path}")
    
    # 获取数据
    images = zarr_data['data/left_wrist_img']
    external_images = zarr_data['data/external_img'] if 'data/external_img' in zarr_data else zarr_data['data/left_wrist_img']
    tcp_poses = zarr_data['data/left_robot_tcp_pose']
    actions = zarr_data['data/action']
    
    # 获取episode_ends，如果存在
    episode_ends = None
    if 'meta/episode_ends' in zarr_data:
        episode_ends = zarr_data['meta/episode_ends'][:]
    
    # 创建data子文件夹（如果不存在）
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 清理输出目录
    output_path = data_dir / output_repo
    if output_path.exists():
        print(f"清理已存在的输出目录: {output_path}")
        shutil.rmtree(output_path)
    
    # 清理默认缓存目录
    default_cache_paths = [
        Path(os.path.expanduser("~/.cache/lerobot")) / output_repo,
        Path(os.path.expanduser("~/.cache/huggingface/lerobot")) / output_repo
    ]
    
    for cache_path in default_cache_paths:
        if cache_path.exists():
            print(f"清理已存在的缓存目录: {cache_path}")
            shutil.rmtree(cache_path)
    
    # 创建LeRobot数据集
    print(f"创建LeRobot数据集: {output_repo}")
    dataset = LeRobotDataset.create(
        repo_id=output_repo,
        robot_type="flexiv",  
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (9,),  # TCP姿态有9个维度
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (10,),  # 动作有10个维度
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
 
    if episode_ends is not None:
   
        episode_boundaries = [0] + list(episode_ends)
    else:

        episode_boundaries = [0, len(images)]
    
    # 处理每个episode
    print(f"开始处理数据，共 {len(images)} 帧，分为 {len(episode_boundaries)-1} 个episode")
    for ep_idx in range(len(episode_boundaries) - 1):
        start_idx = episode_boundaries[ep_idx]
        end_idx = episode_boundaries[ep_idx + 1]
        
        print(f"处理Episode {ep_idx+1}/{len(episode_boundaries)-1}，帧范围: {start_idx}-{end_idx}")
        
        # 处理每一帧
        for frame_idx in tqdm(range(start_idx, end_idx), desc=f"Episode {ep_idx+1}"):
            # 获取图像数据
            wrist_img = images[frame_idx]
            external_img = external_images[frame_idx] if external_images is not None else wrist_img
            
            # 调整图像大小到256x256
            wrist_img_resized = cv2.resize(wrist_img, (256, 256))
            external_img_resized = cv2.resize(external_img, (256, 256))
            
            # 获取状态和动作数据
            state = tcp_poses[frame_idx]
            action = actions[frame_idx]
            
            # 添加帧到数据集
            dataset.add_frame(
                {
                    "image": external_img_resized,
                    "wrist_image": wrist_img_resized,
                    "state": state,
                    "actions": action,
                }
            )
        
        # 保存episode
        dataset.save_episode(task=task_description)
    
    # 整合数据集
    print("整合数据集")
    dataset.consolidate(run_compute_stats=False)
    
    # 可选：推送到Hugging Face Hub
    if push_to_hub:
        print(f"将数据集推送到Hugging Face Hub: {output_repo}")
        dataset.push_to_hub(
            tags=["robot", "panda", "zarr"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
    
    for cache_path in default_cache_paths:
        if cache_path.exists():
            print(f"将数据从默认位置 {cache_path} 复制到 {output_path}")
            if not output_path.exists():
                try:
                    shutil.copytree(cache_path, output_path)
                    print(f"复制完成: {cache_path} -> {output_path}")
                    break  
                except Exception as e:
                    print(f"复制过程中出错: {e}")
    
    # 打印数据位置信息
    print("\n数据位置信息:")
    for cache_path in default_cache_paths:
        if cache_path.exists():
            print(f"- 缓存位置: {cache_path} (存在)")
            print(f"  包含文件: {os.listdir(cache_path)}")
        else:
            print(f"- 缓存位置: {cache_path} (不存在)")
    
    if output_path.exists():
        print(f"- 输出位置: {output_path} (存在)")
        print(f"  包含文件: {os.listdir(output_path)}")
    else:
        print(f"- 输出位置: {output_path} (不存在)")
    
    print(f"\n转换完成！数据集已保存到: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='将Zarr数据转换为LeRobot格式')
    parser.add_argument('--zarr_path', required=True, help='Zarr数据的路径')
    parser.add_argument('--output_repo', required=True, help='输出数据集的仓库名称')
    parser.add_argument('--fps', type=int, default=30, help='数据集的帧率，默认为30')
    parser.add_argument('--task_description', default="Grasp and place an object.", help='任务描述')
    parser.add_argument('--push_to_hub', action='store_true', help='是否将数据集推送到Hugging Face Hub')
    
    args = parser.parse_args()
    
    convert_zarr_to_lerobot(
        args.zarr_path,
        args.output_repo,
        args.fps,
        args.task_description,
        args.push_to_hub
    )


if __name__ == "__main__":
    main() 
