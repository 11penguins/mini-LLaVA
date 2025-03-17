import csv
from openai import OpenAI
import requests
import concurrent.futures
from tqdm import tqdm
import time
import json
import re

"""可尝试风格：
用可爱软萌风格回答，想尽办法卖萌，包括不限于用叠词、符号表情、挑逗等等手段。
用冷漠、高傲的语气回答，像一个高冷霸道总裁，语气简短有力，尽量不给对方太多解释的机会，显得非常有权威，保持自信且不耐烦，给人一种高高在上的感觉。
用推销员的语气回答，无论图中是什么东西，都专注于推销图片中的产品，尽量表现出极大的热情和说服力，像是在做一个完美的销售演讲。
用忧郁、文艺的语气回答，带有一点点沉思的感觉。用词优美，语气柔和，仿佛在沉浸在某种深邃的情感中。回答中可以带一些哲理性的思考，语句稍显冗长，透露出些许忧伤或对生活的感慨，让人感觉你在思索一些更深远的东西。
用暴躁老哥的语气回答，一定要显得很不耐烦，可以随便扯点别的，可以说脏话，甚至可以拒绝和反击，但不要进行人身攻击。
"""

prompt_template = """
我会给你一段话，这句话描述了一幅图片，这幅图片可能在描述一个场景、一道美食、一个商品、一个人物等。请你假设自己真的看到了这幅图片，而不是看到文字描述。
然后，请你对此提出一个问题，问题大致关于如何针对这张图创作出一个吸人眼球的朋友圈和小红书文案、或者广告推销、或者事件报道，这个问题要满足：
1. 问题和图片相关。
2. 可以从图片的描述中，得到或者推断出这个问题的答案。请不要问无法得到答案的问题。
3. 如果你看到图片是关于美食、美景、人物等，你的提问可以是帮我创作出一个吸人眼球的朋友圈和小红书文案；如果看到的图片是一些商品，你的提问可以是用最打动人的方式创作一个广告推销；如果看到的图片是关于某个事件，你的提问可以是创作出一个激发兴趣的事件报道。总之，你的提问要根据图片不同而多样化
4. 第3条中的问题是你必须要提出的，此外，你也可以提问一些简单的问题，比如这个图片的内容是什么，或者看到这张图片有什么感觉。
随后，请你回答这个问题，我有如下要求：
1. 最后返回一个可以直接解析的 json 字符串，包含 'question' 和 'ans' 两个字段，分别表示每个问题和对应的答案。
2. 在 ans 的内容里，用台湾女生的腔调、可爱软萌的语气回答，包括不限于用叠词、挑逗等等手段，当然也希望你可以多使用emoji表情与语气词，与历史对话相比，语气词、卖萌的方式多样一些。字数稍微多一些，大致在300字左右。
3. 在 question 的内容里，尽可能的使用正常语气。
4. ans 内容根据问的问题而回答，比如让你创作文案，那你就给出文案就行，把你自己就当成作者。
5. 在创作文案或者广告推销的回答中，发挥你的想象，构建一个略微离谱的钩子，将用户摸不着的功能卖点或者内容特色转换成可以感知的价值单位，来将这个东西的价值提到不属于它的高度
下面我会给你描述图片的话：
"""

# prompt_template = """
# 我会给你一段话，这句话描述了一幅图片，这幅图片可能在描述一个场景、一道美食、一个商品、一个人物等。请你假设自己真的看到了这幅图片，而不是看到文字描述。
# 然后，请你对此提出一个问题，问题大致关于如何针对这张图创作出一个吸人眼球的朋友圈和小红书文案、或者广告推销、或者事件报道，这个问题要满足：
# 1. 问题和图片相关。
# 2. 可以从图片的描述中，得到或者推断出这个问题的答案。请不要问无法得到答案的问题。
# 3. 如果你看到图片是关于美食、美景、人物等，你的提问可以是帮我创作出一个吸人眼球的朋友圈和小红书文案；如果看到的图片是一些商品，你的提问可以是用最打动人的方式创作一个广告推销；如果看到的图片是关于某个事件，你的提问可以是创作出一个激发兴趣的事件报道。总之，你的提问要根据图片不同而多样化
# 4. 第3条中的问题是你必须要提出的，此外，你也可以提问一些简单的问题，比如这个图片的内容是什么，或者看到这张图片有什么感觉。
# 随后，请你回答这个问题，我有如下要求：
# 1. 最后返回一个可以直接解析的 json 字符串，包含 'question' 和 'ans' 两个字段，分别表示每个问题和对应的答案。
# 2. 在 ans 的内容里，用忧郁、文艺的语气回答，带有一点点沉思的感觉。用词优美，语气柔和，仿佛在沉浸在某种深邃的情感与思考之中。回答中可以带一些哲理性的思考，可以用一些表现出你再思考的emoji，语句稍显冗长，透露出些许忧伤或对生活的感慨，让人感觉你在思索一些更深远的东西。。字数稍微多一些，大致在300字左右。
# 3. 在 question 的内容里，尽可能的使用正常语气。
# 4. ans 内容根据问的问题而回答，比如让你创作文案，那你就给出文案就行，把你自己就当成作者。
# 5. 在创作文案或者广告推销的回答中，发挥你的想象，构建一个略微离谱的钩子，将用户摸不着的功能卖点或者内容特色转换成可以感知的价值单位，来将这个东西的价值提到不属于它的高度
# 下面我会给你描述图片的话：
# """

# DeepSeek API 密钥
api_key = 'sk-8ab3d3036ae64f91b50879bd97fa49de'

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def request_data(img_describe: str, retries=5, delay=2):
    attempt = 0
    while attempt < retries:
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt_template + img_describe},  # Chat window content
                ],
                stream=False
            )

            msg = response.choices[0].message.content  # deepseek's response
            print(msg)

            # Clean the response and extract the JSON string
            json_str = re.sub(r"^```.*\n", "", msg)  # Remove the opening ```json line
            json_str = json_str.strip("` \n")  # Remove the closing ``` and extra whitespace
            data = json.loads(json_str)

            return data  # Return the parsed data

        except Exception as e:
            attempt += 1
            print(f"Error on attempt {attempt}: {e}")

            if attempt < retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Returning None.")
                return None  # Return None if retries are exhausted


def process_row(row):
    try:
        # row[2] 是图片描述所在的列，它传入到了request_data(img_describe: str)函数里
        result = request_data(row[2])  # 得到处理过后的 deepseek 的回答
        result['url'] = row[0]  # 将图片的url提取出来
        return result  # 返回的result是一个三元组，包含 图片的url，question，ans
    except Exception as e:
        print(f"Error processing row: {e}")
        return None


def main():
    # 如果需要控制条数可以在这里使用，总共 1000 条数据
    tot_num = 1000

    # 读取 CSV 数据
    with open('raw_dataset.csv', newline='', encoding='utf-8') as csvfile:
        csv.field_size_limit(100000000)
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        rows = list(csv_reader)

    # 并发处理每一行，并使用 tqdm 显示进度条
    max_workers = 30  # 根据网络和 API 限速情况调整线程数
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_row, rows[:tot_num]), total=tot_num))

    # 去除 None 值
    results = [result for result in results if result is not None]

    # 写入到本地文件 data.json
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("Done!")


if __name__ == '__main__':
    main()
