#!/usr/bin/env python3
"""
快速开始脚本 - 一键运行药物疾病关系预测
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description):
    """运行命令并处理错误"""
    print(f"\n{'='*50}")
    print(f"正在执行: {description}")
    print(f"命令: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ 成功完成")
        if result.stdout:
            print("输出:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 执行失败: {e}")
        if e.stdout:
            print("标准输出:", e.stdout)
        if e.stderr:
            print("错误输出:", e.stderr)
        return False


def setup_environment():
    """设置环境"""
    print("🔧 设置Python环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ 需要Python 3.8或更高版本")
        return False
    
    print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 创建必要的目录
    directories = [
        "data/raw",
        "data/processed", 
        "data/splits",
        "logs",
        "checkpoints",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {directory}")
    
    return True


def install_dependencies():
    """安装依赖包"""
    print("📦 安装依赖包...")

    # 使用专门的安装脚本
    success = run_command(
        "python install_dependencies.py",
        "安装所有依赖包（包括PyTorch Geometric）"
    )

    if success:
        print("✅ 依赖包安装完成")
    else:
        print("⚠️ 自动安装失败，尝试手动安装基础依赖...")
        # 回退到基础安装
        fallback_success = run_command(
            "pip install torch pandas numpy scikit-learn matplotlib seaborn tqdm pyyaml requests jupyter",
            "安装基础依赖包"
        )
        if fallback_success:
            print("✅ 基础依赖安装完成，PyTorch Geometric需要手动安装")
            print("💡 手动安装命令:")
            print("   pip install torch-geometric torch-scatter torch-sparse torch-cluster")
        return fallback_success

    return success


def download_data():
    """下载PrimeKG数据"""
    print("📥 下载PrimeKG数据集...")
    
    success = run_command(
        "python src/data_loader.py --download",
        "下载PrimeKG数据集"
    )
    
    if success:
        print("✅ 数据下载完成")
    
    return success


def process_data():
    """处理数据"""
    print("⚙️ 处理数据...")
    
    success = run_command(
        "python src/data_loader.py --process",
        "处理PrimeKG数据"
    )
    
    if success:
        print("✅ 数据处理完成")
    
    return success


def train_model():
    """训练模型"""
    print("🚀 开始训练模型...")
    
    success = run_command(
        "python src/train.py --config configs/config.yaml",
        "训练RGCN模型"
    )
    
    if success:
        print("✅ 模型训练完成")
    
    return success


def run_data_exploration():
    """运行数据探索"""
    print("📊 启动数据探索...")
    
    success = run_command(
        "jupyter notebook notebooks/data_exploration.ipynb",
        "启动Jupyter Notebook进行数据探索"
    )
    
    return success


def main():
    parser = argparse.ArgumentParser(description="药物疾病关系预测快速开始脚本")
    parser.add_argument("--step", choices=[
        "setup", "install", "download", "process", "train", "explore", "all"
    ], default="all", help="选择执行的步骤")
    parser.add_argument("--skip-install", action="store_true", help="跳过依赖安装")
    parser.add_argument("--skip-download", action="store_true", help="跳过数据下载")
    
    args = parser.parse_args()
    
    print("🎯 药物疾病关系预测 - 快速开始")
    print("=" * 60)
    
    steps_success = []
    
    if args.step in ["setup", "all"]:
        success = setup_environment()
        steps_success.append(("环境设置", success))
        if not success:
            print("❌ 环境设置失败，退出")
            return
    
    if args.step in ["install", "all"] and not args.skip_install:
        success = install_dependencies()
        steps_success.append(("依赖安装", success))
        if not success:
            print("❌ 依赖安装失败，退出")
            return
    
    if args.step in ["download", "all"] and not args.skip_download:
        success = download_data()
        steps_success.append(("数据下载", success))
        if not success:
            print("❌ 数据下载失败，退出")
            return
    
    if args.step in ["process", "all"]:
        success = process_data()
        steps_success.append(("数据处理", success))
        if not success:
            print("❌ 数据处理失败，退出")
            return
    
    if args.step in ["train", "all"]:
        success = train_model()
        steps_success.append(("模型训练", success))
        if not success:
            print("⚠️ 模型训练失败，但可以继续其他步骤")
    
    if args.step == "explore":
        success = run_data_exploration()
        steps_success.append(("数据探索", success))
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 执行总结:")
    print("=" * 60)
    
    for step_name, success in steps_success:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{step_name}: {status}")
    
    all_success = all(success for _, success in steps_success)
    
    if all_success:
        print("\n🎉 所有步骤执行成功！")
        print("\n📝 接下来你可以:")
        print("1. 查看训练日志: logs/")
        print("2. 检查模型检查点: checkpoints/")
        print("3. 运行数据探索: python quick_start.py --step explore")
        print("4. 修改配置文件: configs/config.yaml")
        print("5. 重新训练模型: python src/train.py")
    else:
        print("\n⚠️ 部分步骤执行失败，请检查错误信息")
    
    print("\n📚 更多信息请查看 README.md")


if __name__ == "__main__":
    main()
