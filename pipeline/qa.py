from typing import Iterable
import jsonlines


def read_jsonl(path):
    content = []
    with jsonlines.open(path, "r") as json_file:
        for obj in json_file.iter(type=dict, skip_invalid=True):
            content.append(obj)
    return content


def save_answers(
        queries: Iterable, results: Iterable, path: str = "data/answers.jsonl"
):
    answers = []
    Reanswers = []
    for query, result in zip(queries, results):
        Reanswers.append(
            {"id": query["id"], "query": query["query"], "answer": result[0].text, "context": result[1]}
        )
        answers.append(
            {"id": query["id"], "query": query["query"], "answer": result[0].text}
        )

    # use jsonlines to save the answers
    def write_jsonl(path, content):
        with jsonlines.open(path, "w") as json_file:
            json_file.write_all(content)

    # 保存答案到 data/answers.jsonl
    write_jsonl(path, answers)
    write_jsonl("submit_result_with_context_and_summary.jsonl", Reanswers)


def getFullName(log_file_path):
    global lines
    # 使用with语句安全地打开文件
    with open(log_file_path, 'r', encoding='utf-8') as file:
        # 读取所有行到一个列表中
        lines = file.readlines()

    # 创建一个新列表，用于存储没有空行的行
    non_blank_lines = [line.strip() for line in lines if line.strip()]

    logWords = {}
    suolue = ""
    suols = set()
    wanzs = set()
    for line in non_blank_lines:
        if len(line) < 10:
            if len(wanzs):
                wanzslist = list(wanzs)
                logWords[suolue] = wanzslist
                wanzs.clear()
            if line in suols:
                continue
            suols.add(line)
            suolue = line
        else:
            wanzs.add(line)
    return logWords
