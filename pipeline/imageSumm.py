import os
from bs4 import BeautifulSoup
import urllib.parse
from zhipuai import ZhipuAI

import base64


def png_to_base64(file_path):
    try:
        with open(file_path, 'rb') as image_file:
            # 读取PNG图像文件的内容
            image_data = image_file.read()
            # 使用base64编码
            base64_encoded = base64.b64encode(image_data)
            # 转换为字符串并返回
            return base64_encoded.decode('utf-8')
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def split_path(path):
    # 将路径按照分隔符切分
    parts = path.split('\\')

    # 保留最后三项地址
    last_three_parts = parts[-3:]

    return last_three_parts


def count_images_and_tables(html_file):
    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        # 统计图像数量
        images = soup.find_all('img')
        num_images = len(images)
        image_info = []
        if images:
            # 如果存在图像，则输出图像的信息
            for img in images:
                src = img.get('src')
                image_info.append(src)
        # 统计表格数量
        tables = soup.find_all('table')
        num_tables = len(tables)
        return num_images, num_tables, image_info


def image_summary(config, file_paths):
    context = "\n"
    rootPath = "D:\\MyPyCharm\\LLMTuning\\aiops24-RAG-demo-glm\\demo\\dataset\\"
    image_paths = []
    for path in file_paths:
        splited_path = split_path(path)
        full_html_path = rootPath + splited_path[0] + "\\documents\\" + splited_path[1] + "\\topics\\" + \
                         splited_path[2].split('.')[0] + ".html"
        if not os.path.exists(full_html_path): # 网址找不到
            continue
        num_images, num_tables, image_infoes = count_images_and_tables(full_html_path)
        print(f"Number of images: {num_images}")
        print(f"Number of tables: {num_tables}")

        if not num_images:  # 图片找不到
            continue

        index1 = full_html_path.rfind("\\")
        for image_info in image_infoes:
            if not image_info:
                continue
            image_path = full_html_path[:index1] + "\\" + image_info
            decoded_str = urllib.parse.unquote(image_path, encoding='utf-8')
            image_paths.append(decoded_str)
            print(decoded_str)
    for image_path in image_paths:
        base64_image = png_to_base64(image_path)  # 生成base64的图像
        client = ZhipuAI(api_key=config["GLM_KEY"])
        response = client.chat.completions.create(
            model="glm-4v",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请对图中的数据生成摘要"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image
                            }
                        }
                    ]
                }
            ],
        )
        print(response.choices[0].message.content)
        context = context + response.choices[0].message.content
    return context