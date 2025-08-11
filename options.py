import argparse
import os
from pathlib import Path


# ==============================================================================
# 这是一个假设的 main 函数，因为您的原始代码中调用了它，但没有提供其定义
# 请确保您的文件中存在类似这样的主训练函数
# ==============================================================================
def main(args):
    """
    模型训练的主函数。
    """
    print("-----> model save name:", args.model_name, "<-----")

    # 根据 local 参数选择数据根目录
    # 【已修改】这里的逻辑确保了会根据运行环境选择正确的路径
    data_root = args.root_local if args.local == "True" else args.root_sever

    # 示例：打印将要使用的数据集路径
    print(f"数据根目录 (Data Root): {data_root}")
    sketch_path = Path(data_root) / "sketch_png"
    print(f"草图路径 (Sketch Path): {sketch_path}")

    # ==================================================================
    # 您实际的训练逻辑应该在这里开始
    # 例如：
    # 1. 创建数据集和数据加载器
    # 2. 初始化模型
    # 3. 设置优化器
    # 4. 循环进行训练和验证
    # ...
    # 以下为占位代码，请替换为您自己的训练代码
    if not sketch_path.exists():
        print(
            f"错误：找不到草图目录 {sketch_path}。请检查您的 --root_local 或 --root_sever 参数是否正确。"
        )
        return

    print("\n[INFO] 训练逻辑开始... (此处为占位符)")
    # ... 您的训练代码 ...
    print("[INFO] 训练完成。")
    # ==================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="草图检索模型训练脚本")

    # --- 基本设置 ---
    parser.add_argument("--bs", type=int, default=32, help="批次大小 (Batch Size)")
    parser.add_argument(
        "--embed_dim", type=int, default=512, help="嵌入维度 (Embedding Dimension)"
    )
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载进程数")
    parser.add_argument(
        "--weight_dir", type=str, default="model_trained", help="模型权重保存目录"
    )

    # --- 模型架构 ---
    parser.add_argument(
        "--sketch_model",
        type=str,
        default="vit",
        choices=["vit", "lstm", "sdgraph"],
        help="草图编码器模型",
    )
    parser.add_argument(
        "--image_model", type=str, default="vit", choices=["vit"], help="图像编码器模型"
    )

    # --- 任务设置 ---
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="cl",
        choices=["cl", "fg"],
        help="检索级别: cl (类别级), fg (细粒度)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="zs_sbir",
        choices=["sbir", "zs_sbir"],
        help="任务类型: sbir (标准), zs_sbir (零样本)",
    )
    parser.add_argument(
        "--pair_mode",
        type=str,
        default="single_pair",
        choices=["multi_pair", "single_pair"],
        help="草图-图像配对模式",
    )

    # --- 路径设置 ---
    # 【关键修改】将 --local 的默认值从字符串 "True" 改为布尔值 True，并使用 action='store_true' 使其成为一个开关
    parser.add_argument(
        "--local",
        action="store_true",
        help="设置为本地运行模式 (默认不设置则为服务器模式)",
    )

    # 【关键修改】为服务器和本地环境提供了正确的默认路径
    parser.add_argument(
        "--root_sever",
        type=str,
        default="/opt/data/private/data_set/sketch_retrieval",
        help="服务器上的数据根目录",
    )
    # 【关键修改】将本地路径从 Windows 格式改为 WSL 格式
    parser.add_argument(
        "--root_local",
        type=str,
        default="/mnt/c/Users/grfpa/Desktop/sketchy",
        help="本地 (WSL) 环境下的数据根目录",
    )

    # --- 训练策略 ---
    parser.add_argument("--epoch", type=int, default=1000, help="最大训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    # 【已修改】将字符串选项改为布尔开关，更符合常规用法
    parser.add_argument(
        "--freeze_image_encoder", action="store_true", help="冻结图像编码器"
    )
    parser.add_argument(
        "--no-freeze_sketch_backbone",
        dest="freeze_sketch_backbone",
        action="store_false",
        help="不冻结草图编码器主干",
    )
    parser.add_argument(
        "--no-create_fix_data_file",
        dest="create_fix_data_file",
        action="store_false",
        help="不创建固定数据集划分文件",
    )
    parser.add_argument(
        "--no-load_ckpt", dest="load_ckpt", action="store_false", help="不加载检查点"
    )

    # --- 其他 ---
    parser.add_argument(
        "--add_str", type=str, default="", help="附加到模型名称的字符串"
    )
    parser.add_argument(
        "--vis", action="store_true", help="是否可视化草图特征 (此模式下不训练)"
    )
    parser.add_argument(
        "--full_train", action="store_true", help="是否使用全部数据训练"
    )

    args = parser.parse_args()

    # 根据参数动态生成模型名称
    args.model_name = f"{args.sketch_model}_{args.image_model}_{args.retrieval_mode}_{args.task}_{args.pair_mode}"
    if args.add_str:
        args.model_name += f"_{args.add_str}"

    # 设置一些默认值为 True 的布尔参数
    parser.set_defaults(
        freeze_sketch_backbone=True, create_fix_data_file=True, load_ckpt=True
    )
    args = parser.parse_args()

    return args


# 【关键修改】整理了脚本的入口点
if __name__ == "__main__":

    # 1. 解析命令行参数
    # 之前这部分代码被放在了脚本的顶层，现在将其放入主入口中，这是标准做法。
    options = parse_args()

    # 2. 调用主训练函数
    # 之前这行调用在 if __name__ == "__main__" 之外，现在移到这里。
    # 您需要确保您的文件中有名为 `main` 的函数来执行训练。
    # 由于您的代码片段中没有 main 函数的定义，我在这里创建了一个占位函数。
    main(options)

    # ==============================================================================
    # 【已注释掉】以下是您脚本末尾的无关代码块。
    # 这段代码的功能是计算两个文件的对称差，和模型训练无关。
    # 将其放在训练脚本中会引起混淆，并因为它使用了硬编码的Windows路径，在WSL中运行时会失败。
    # 如果您需要这个功能，建议将其保存为一个单独的Python文件（例如 `process_files.py`）。
    # ==============================================================================
    # print("\n--- 开始执行独立的文件处理任务 ---")
    # f1, f2, fout = (
    #     r"C:\Users\ChengXi\Desktop\60mm20250708\acc-1.txt",
    #     r"C:\Users\ChengXi\Desktop\60mm20250708\acc-5.txt",
    #     r"C:\Users\ChengXi\Desktop\60mm20250708\acc-5-filter.txt",
    # )
    #
    # try:
    #     with open(f1, "r", encoding="utf-8") as fp:
    #         set1 = set(line.rstrip("\n") for line in fp)
    #
    #     with open(f2, "r", encoding="utf-8") as fp:
    #         set2 = set(line.rstrip("\n") for line in fp)
    #
    #     # 并集 - 交集 ＝ 对称差
    #     sym_diff = sorted((set1 | set2) - (set1 & set2))
    #
    #     with open(fout, "w", encoding="utf-8") as fp:
    #         for line in sym_diff:
    #             fp.write(line + "\n")
    #
    #     print(f"对称差已写入 {fout}，共 {len(sym_diff)} 行。")
    # except FileNotFoundError:
    #     print(f"错误：找不到文件，请检查路径。此任务使用的是硬编码的Windows路径。")
    # print("--- 文件处理任务结束 ---")
