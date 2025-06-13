import cv2
import numpy as np
from scipy.optimize import minimize
import time
from PIL import Image, ImageDraw, ImageFont

# ==============================================================================
# --- 全局配置参数 (Global Configuration) ---
# 可以在此部分修改所有关键参数，而无需改动后续的函数代码。
# ==============================================================================

# 1. 文件路径配置
# 图a: 参考地图 (Ground Truth)
REF_MAP_PATH = 'map1.png'
# 图b: SLAM建图结果
SLAM_MAP_PATH = 'hector-fix.png'

# 2. 图像预处理参数
# 用于将灰度图转换为纯黑白二值图的亮度阈值
BINARY_THRESHOLD = 128

# 3. 边界模糊（膨胀）参数
# 用于拓宽参考地图边界的模糊核大小。值越大，容错空间越大。
# 1: 不进行模糊
# 3: 拓宽1个像素的边界
# 5: 拓宽2个像素的边界 (注意: 必须是正奇数)
BLUR_KERNEL_SIZE = 3

# 4. 优化算法参数
# 定义自动对齐时，算法对旋转、缩放、平移的搜索范围
# 角度搜索范围 (度)
ANGLE_BOUNDS = (-15, 15)
# 缩放比例搜索范围
SCALE_BOUNDS = (0.8, 1.2)
# 平移像素搜索范围 (x和y方向相同)
TRANSLATION_BOUNDS = (-100, 100)
# 优化器的最大迭代次数，增加此值可以提高精度，但会增加计算时间
MAX_ITERATIONS = 500

# 5. 最终得分权重配置
# 定义像素重叠相似度和轮廓形状相似度在最终综合得分中的比重。
# 两者总和应为1.0。
PIXEL_SIMILARITY_WEIGHT = 0.4  # 像素重叠度占60%
SHAPE_SIMILARITY_WEIGHT = 0.6  # 形状相似度占40%


# ==============================================================================
# --- 核心算法函数 (Core Algorithm Functions) ---
# 一般情况下，您无需修改此部分的代码。
# ==============================================================================

def preprocess_image(image_path, threshold):
    """
    加载图像文件，并将其转换为二值化的Numpy数组。

    Args:
        image_path (str): 图像文件路径。
        threshold (int): 二值化亮度阈值。

    Returns:
        numpy.ndarray: 二值化后的图像数组 (1代表障碍物, 0代表自由空间)。
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法找到或打开图片: {image_path}")
    # 使用THRESH_BINARY_INV，使得低于阈值的像素（如黑色）变为255，高于则为0
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    # 将255归一化为1，便于计算
    return (binary_img / 255).astype(np.uint8)


def dilate_reference_map(ref_map, kernel_size):
    """
    对参考地图的障碍物边界进行膨胀（拓宽），为对齐提供容错空间。

    Args:
        ref_map (numpy.ndarray): 参考地图的二值化数组。
        kernel_size (int): 膨胀操作的核大小。

    Returns:
        numpy.ndarray: 膨胀处理后的参考地图。
    """
    if kernel_size <= 1:
        return ref_map
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(ref_map, kernel, iterations=1)


def calculate_pixel_similarity(intersection_ref_map, transformed_map):
    """
    计算像素重叠相似度（精确率）。
    公式: (与参考图重叠的SLAM图像素) / (SLAM图的总像素)

    Args:
        intersection_ref_map (numpy.ndarray): 用于计算交集的参考地图（可能已膨胀）。
        transformed_map (numpy.ndarray): 对齐后的SLAM地图。

    Returns:
        float: 像素重叠相似度得分。
    """
    if intersection_ref_map.shape != transformed_map.shape:
        h, w = intersection_ref_map.shape
        transformed_map = cv2.resize(transformed_map, (w, h), interpolation=cv2.INTER_NEAREST)

    intersection = np.sum(np.logical_and(intersection_ref_map == 1, transformed_map == 1))
    total_black_pixels_in_slam = np.sum(transformed_map)

    if total_black_pixels_in_slam == 0:
        return 0.0

    return intersection / total_black_pixels_in_slam


def calculate_shape_similarity(ref_map, aligned_map):
    """
    使用OpenCV的轮廓匹配功能，计算两个地图的形状相似度。

    Args:
        ref_map (numpy.ndarray): 原始（未膨胀）的参考地图。
        aligned_map (numpy.ndarray): 对齐后的SLAM地图。

    Returns:
        float: 形状相似度得分 (1.0为完美匹配)。
    """
    contours_ref, _ = cv2.findContours((ref_map * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_aligned, _ = cv2.findContours((aligned_map * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    if not contours_ref or not contours_aligned:
        print("警告: 未能在其中一张或两张图片中找到有效的轮廓用于形状比较。")
        return 0.0

    main_contour_ref = max(contours_ref, key=cv2.contourArea)
    main_contour_aligned = max(contours_aligned, key=cv2.contourArea)

    match_value = cv2.matchShapes(main_contour_ref, main_contour_aligned, cv2.CONTOURS_MATCH_I1, 0.0)
    # 将返回值归一化为0到1之间的相似度得分，值越小形状越像
    return 1.0 / (1.0 + match_value)


def find_best_match(ref_map, slam_map, blur_kernel_size):
    """
    核心函数：通过优化算法自动寻找最佳的变换参数（旋转、缩放、平移），
    以使得SLAM地图与参考地图的像素重叠度最大化。
    """
    print("开始进行地图相似度分析...")
    start_time = time.time()

    # 创建一个膨胀版的参考地图，仅用于优化过程中的相似度计算，以提供容错
    intersection_ref_map = dilate_reference_map(ref_map, kernel_size=blur_kernel_size)
    print(f"参考地图已进行边界拓宽，核大小: {blur_kernel_size}x{blur_kernel_size}")

    h, w = intersection_ref_map.shape

    # 定义优化器需要最小化的目标函数
    def objective_function(params):
        angle, scale, tx, ty = params
        center = (slam_map.shape[1] // 2, slam_map.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        transformed_slam_map = cv2.warpAffine(slam_map, M, (w, h), flags=cv2.INTER_NEAREST)

        # 优化器以最大化像素重叠度为目标
        score = calculate_pixel_similarity(intersection_ref_map, transformed_slam_map)
        # minimize函数是求最小值，因此返回 1.0 - score
        return 1.0 - score

    initial_guess = [0.0, 1.0, 0.0, 0.0]
    bounds = [ANGLE_BOUNDS, SCALE_BOUNDS, TRANSLATION_BOUNDS, TRANSLATION_BOUNDS]

    print("正在通过优化算法搜索最佳匹配参数...")
    result = minimize(
        objective_function,
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': MAX_ITERATIONS, 'disp': True}
    )

    end_time = time.time()
    print(f"优化过程完成，耗时: {end_time - start_time:.2f} 秒")

    best_params = result.x
    best_pixel_score = 1.0 - result.fun

    # 使用找到的最佳参数生成最终对齐的地图
    best_angle, best_scale, best_tx, best_ty = best_params
    center = (slam_map.shape[1] // 2, slam_map.shape[0] // 2)
    M_best = cv2.getRotationMatrix2D(center, best_angle, best_scale)
    M_best[0, 2] += best_tx
    M_best[1, 2] += best_ty
    aligned_map = cv2.warpAffine(slam_map, M_best, (w, h), flags=cv2.INTER_NEAREST)

    # 基于最佳对齐结果，计算形状相似度（使用原始参考图）
    best_shape_score = calculate_shape_similarity(ref_map, aligned_map)

    return best_pixel_score, best_shape_score, best_params, aligned_map


def visualize_results(ref_map, aligned_map, best_pixel_score, best_shape_score, final_weighted_score, best_params):
    """
    生成并显示最终的对比结果图。
    - 背景: 白色
    - 重叠部分 (正确): 绿色
    - 未重叠部分 (误差): 红色
    """
    # 1. 创建一个纯白色的背景画布
    h, w = ref_map.shape
    visualization_img = np.ones((h, w, 3), dtype=np.uint8) * 255

    # 2. 找出各个区域
    true_positives = np.logical_and(ref_map == 1, aligned_map == 1)
    false_positives = np.logical_and(ref_map == 0, aligned_map == 1)
    false_negatives = np.logical_and(ref_map == 1, aligned_map == 0)

    # 3. 在画布上着色
    visualization_img[false_positives] = [0, 0, 255]  # BGR for Red
    visualization_img[false_negatives] = [0, 0, 255]  # BGR for Red
    visualization_img[true_positives] = [0, 255, 0]  # BGR for Green

    # 4. 添加带中文的文本信息和图例
    pil_img = Image.fromarray(cv2.cvtColor(visualization_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    try:
        font_path = "msyh.ttc"
        main_font = ImageFont.truetype(font_path, 28)
        sub_font = ImageFont.truetype(font_path, 20)
        legend_font = ImageFont.truetype(font_path, 20)
        print(f"成功加载字体: {font_path}")
    except IOError:
        print(f"警告: 未在指定路径找到'msyh.ttc'。正在尝试备用字体...")
        try:
            font_path_alt = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
            main_font = ImageFont.truetype(font_path_alt, 24)
            sub_font = ImageFont.truetype(font_path_alt, 18)
            legend_font = ImageFont.truetype(font_path_alt, 16)
            print(f"成功加载备用字体: {font_path_alt}")
        except IOError:
            print("警告: 未找到任何中文字体，将使用默认字体。中文可能无法正确显示。")
            main_font = ImageFont.load_default()
            sub_font = ImageFont.load_default()
            legend_font = ImageFont.load_default()

    # --- 绘制得分文本 (左上角) ---
    blue_color = (65, 105, 225)  # 皇家蓝
    draw.text((30, 30), f"综合得分: {final_weighted_score:.4f}", font=main_font, fill=blue_color)
    draw.text((30, 65), f"像素重叠得分: {best_pixel_score:.4f} (权重: {PIXEL_SIMILARITY_WEIGHT})", font=sub_font,
              fill=blue_color)
    draw.text((30, 95), f"轮廓形状得分: {best_shape_score:.4f} (权重: {SHAPE_SIMILARITY_WEIGHT})", font=sub_font,
              fill=blue_color)

    # --- 绘制图例 (中间正上方) ---
    legend_y = 40
    box_size = 35
    text_offset_x = 5
    item_padding = 20

    # 定义图例项
    legend_items = [
        {"color": (0, 255, 0), "text": "重叠部分 "},
        {"color": (255, 0, 0), "text": "未重叠部分"}
    ]

    # 计算图例总宽度以便居中
    total_legend_width = 0
    item_widths = []
    for item in legend_items:
        # 使用 textlength 获取文本宽度
        text_width = draw.textlength(item["text"], font=legend_font)
        item_width = box_size + text_offset_x + text_width
        item_widths.append(item_width)
        total_legend_width += item_width
    total_legend_width += item_padding * (len(legend_items) - 1)

    # 计算起始X坐标
    current_x = (w - total_legend_width) / 2

    # 绘制每个图例项
    for i, item in enumerate(legend_items):
        # 绘制颜色方块
        draw.rectangle(
            [(current_x, legend_y), (current_x + box_size, legend_y + box_size)],
            fill=item["color"]
        )
        # 绘制文本
        draw.text(
            (current_x + box_size + text_offset_x, legend_y),
            item["text"],
            font=legend_font,
            fill=blue_color
        )
        # 更新下一个图例项的起始X坐标
        current_x += item_widths[i] + item_padding

    # 将Pillow图像转换回OpenCV图像以便显示
    final_img_to_show = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    cv2.imshow("Similarity Analysis Result", final_img_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        if not np.isclose(PIXEL_SIMILARITY_WEIGHT + SHAPE_SIMILARITY_WEIGHT, 1.0):
            raise ValueError("错误: 权重总和不为1.0，请检查全局配置参数。")

        ref_map_binary = preprocess_image(REF_MAP_PATH, BINARY_THRESHOLD)
        slam_map_binary = preprocess_image(SLAM_MAP_PATH, BINARY_THRESHOLD)

        pixel_sim, shape_sim, best_parameters, final_aligned_map = find_best_match(
            ref_map_binary,
            slam_map_binary,
            blur_kernel_size=BLUR_KERNEL_SIZE
        )

        weighted_final_score = (pixel_sim * PIXEL_SIMILARITY_WEIGHT) + (shape_sim * SHAPE_SIMILARITY_WEIGHT)

        print("\n--- 分析结果 ---")
        print(f"像素重叠相似度 (精确率): {pixel_sim:.4f}")
        print(f"轮廓形状相似度: {shape_sim:.4f}")
        print("---------------------------------")
        print(f"加权综合相似度: {weighted_final_score:.4f}")
        print("---------------------------------")
        print(f"最佳变换参数 (角度, 缩放, x平移, y平移):")
        print(f"  Angle: {best_parameters[0]:.2f} 度")
        print(f"  Scale: {best_parameters[1]:.2f}")
        print(f"  dx: {best_parameters[2]:.2f} 像素")
        print(f"  dy: {best_parameters[3]:.2f} 像素")

        visualize_results(
            ref_map_binary,
            final_aligned_map,
            pixel_sim,
            shape_sim,
            weighted_final_score,
            best_parameters
        )

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"发生未知错误: {e}")
