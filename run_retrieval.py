import os
import sys
import time
import torch
from qdrant_client import QdrantClient, models
from tqdm import tqdm
import options
from encoders import sbir_model_wrapper
from data.retrieval_datasets import SketchImageDataset, get_split_file_name
from encoders import create_sketch_encoder
from utils import utils
import torchvision.transforms as transforms

# --- 1. 配置 ---
QDRANT_PATH = "./qdrant_db"
IMAGE_COLLECTION_NAME = "sketchy_images_v5"
SKETCH_COLLECTION_NAME = "sketchy_sketches_v5"
MODEL_CHECKPOINT_PATH = "./model_trained/vit_vit_cl_zs_sbir_single_pair.pth"


def load_model_for_inference(args):
    """加载模型并设置为评估模式"""
    print("正在加载模型用于推理...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    model = sbir_model_wrapper.create_sbir_model_wrapper(
        embed_dim=args.embed_dim,
        sketch_model_name=args.sketch_model,
        image_model_name=args.image_model,
    )
    model.to(device)

    try:
        checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"成功从 '{MODEL_CHECKPOINT_PATH}' 加载模型权重。")
    except FileNotFoundError:
        print(f"错误：找不到模型权重文件 '{MODEL_CHECKPOINT_PATH}'。请检查路径。")
        return None, None
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        return None, None

    model.eval()
    return model, device


def setup_qdrant_collection(client, collection_name, vector_size):
    """辅助函数，用于创建或重建Qdrant集合"""
    try:
        client.get_collection(collection_name=collection_name)
        print(f"集合 '{collection_name}' 已存在，将清空并重建。")
    except Exception:
        print(f"集合 '{collection_name}' 不存在，正在创建...")

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size, distance=models.Distance.COSINE
        ),
    )
    print(f"集合 '{collection_name}' 已准备就绪。")


def index_all_data(model, device, args):
    """【索引函数】为数据集中所有的图片和草图生成向量，并分别存入Qdrant。"""
    print("\n--- 开始完整数据索引流程 (图片和草图) ---")
    client = QdrantClient(path=QDRANT_PATH)

    setup_qdrant_collection(client, IMAGE_COLLECTION_NAME, args.embed_dim)
    setup_qdrant_collection(client, SKETCH_COLLECTION_NAME, args.embed_dim)

    print("正在加载数据集以进行完整索引...")
    split_file = get_split_file_name("image", args.pair_mode, args.task)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = SketchImageDataset(
        root=args.root_local,
        mode="test",
        fixed_split_path=split_file,
        sketch_format="image",
        vec_sketch_rep=None,
        sketch_image_subdirs=create_sketch_encoder.get_sketch_info(args.sketch_model)[
            "subdirs"
        ],
        image_transform=transform,
        sketch_transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers
    )

    img_points = []
    skh_points = []
    print("开始生成向量并准备存入 Qdrant...")
    with torch.no_grad():
        for idx_batch, sketches, images, _, category_names in tqdm(
            dataloader, desc="正在索引"
        ):
            images = images.to(device)
            sketches = sketches.to(device)

            image_vectors = model.encode_image(images)
            sketch_vectors = model.encode_sketch(sketches)

            for i in range(len(idx_batch)):
                global_idx = idx_batch[i].item()
                sketch_path, image_path, category = dataset.data_pairs[global_idx]

                img_points.append(
                    models.PointStruct(
                        id=global_idx,
                        vector=image_vectors[i].cpu().numpy(),
                        payload={"path": image_path, "category": category},
                    )
                )
                skh_points.append(
                    models.PointStruct(
                        id=global_idx,
                        vector=sketch_vectors[i].cpu().numpy(),
                        payload={"path": sketch_path, "category": category},
                    )
                )

    if img_points:
        client.upsert(
            collection_name=IMAGE_COLLECTION_NAME, points=img_points, wait=True
        )
    if skh_points:
        client.upsert(
            collection_name=SKETCH_COLLECTION_NAME, points=skh_points, wait=True
        )

    print(f"\n--- 索引完成！---")
    img_info = client.get_collection(collection_name=IMAGE_COLLECTION_NAME)
    skh_info = client.get_collection(collection_name=SKETCH_COLLECTION_NAME)
    print(f"总计 {img_info.points_count} 个图片向量已存入 '{IMAGE_COLLECTION_NAME}'。")
    print(f"总计 {skh_info.points_count} 个草图向量已存入 '{SKETCH_COLLECTION_NAME}'。")


def search_in_collection(client, collection_name, query_vector, k):
    """在指定集合中搜索的辅助函数"""
    start_time = time.perf_counter()
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=k,
        with_payload=True,
    )
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000
    return search_results, duration_ms


def interactive_search(model, device):
    """【主搜索函数】接收用户输入，并返回最相似的图片和草图。"""
    client = QdrantClient(path=QDRANT_PATH)
    print("\n--- 欢迎来到交互式草图检索系统 ---")
    print("输入 'exit' 或 'quit' 来退出程序。")

    while True:
        try:
            sketch_path_input = input("\n请输入要查询的草图文件完整路径: ").strip()
            if sketch_path_input.lower() in ["exit", "quit"]:
                print("程序退出。")
                break

            if not os.path.exists(sketch_path_input):
                if ":\\" in sketch_path_input:
                    parts = sketch_path_input.split(":\\", 1)
                    sanitized_path_part = parts[1].replace("\\", "/")
                    wsl_path = f"/mnt/{parts[0].lower()}/{sanitized_path_part}"

                    if os.path.exists(wsl_path):
                        print(f"路径已自动转换为WSL格式: {wsl_path}")
                        sketch_path_input = wsl_path
                    else:
                        print(
                            f"错误：文件路径不存在 '{sketch_path_input}' (也尝试了WSL路径 '{wsl_path}')"
                        )
                        continue
                else:
                    print(f"错误：文件路径不存在 '{sketch_path_input}'")
                    continue

            # ======================= 关键修改 =======================
            # 使用一个循环来确保获得有效的 K 值
            while True:
                k_input = input("请输入要返回的相似结果数量 (K): ").strip()
                if k_input.isdigit() and int(k_input) > 0:  # 检查是否为数字且大于0
                    k = int(k_input)
                    break  # 输入有效，跳出这个内部循环
                else:
                    print("错误：无效输入，K值必须是一个正整数。请重试。")
            # ======================= 结束修改 =======================

            print(f"\n正在处理查询草图: {os.path.basename(sketch_path_input)}...")
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            sketch_image = utils.image_loader(sketch_path_input, transform)
            sketch_tensor = sketch_image.unsqueeze(0).to(device)

            with torch.no_grad():
                query_vector = (
                    model.encode_sketch(sketch_tensor).cpu().numpy().flatten()
                )

            print("正在搜索相似图片...")
            image_results, img_time = search_in_collection(
                client, IMAGE_COLLECTION_NAME, query_vector, k
            )

            print("正在搜索相似草图...")
            sketch_results, skh_time = search_in_collection(
                client, SKETCH_COLLECTION_NAME, query_vector, k
            )

            print("\n" + "=" * 40)
            print("           *** 检索结果 ***")
            print("=" * 40)

            print(f"\n--- Top {k} 相似图片 (搜索耗时: {img_time:.4f} ms) ---")
            if image_results:
                for i, result in enumerate(image_results):
                    print(
                        f"  {i+1}. (相似度: {result.score:.4f}) | 类别: {result.payload.get('category', 'N/A')}"
                    )
                    print(f"      路径: {result.payload.get('path', 'N/A')}")
            else:
                print("  未找到相似图片。")

            print(f"\n--- Top {k} 相似草图 (搜索耗时: {skh_time:.4f} ms) ---")
            if sketch_results:
                for i, result in enumerate(sketch_results):
                    print(
                        f"  {i+1}. (相似度: {result.score:.4f}) | 类别: {result.payload.get('category', 'N/A')}"
                    )
                    print(f"      路径: {result.payload.get('path', 'N/A')}")
            else:
                print("  未找到相似草图。")

        except Exception as e:
            print(f"在搜索过程中发生未知错误: {e}")


def main():
    should_index = "--index" in sys.argv
    if should_index:
        print("检测到 --index 参数，程序将执行数据库索引。")
        sys.argv.remove("--index")

    model_args = options.parse_args()
    model_args.local = "True"
    model_args.sketch_model = "vit"

    model, device = load_model_for_inference(model_args)

    if not model:
        print("模型加载失败，程序退出。")
        return

    if should_index:
        index_all_data(model, device, model_args)

    interactive_search(model, device)


if __name__ == "__main__":
    main()
